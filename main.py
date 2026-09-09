from __future__ import annotations

import asyncio
import queue
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyaudiowpatch as pyaudio
import soundfile as sf
from mutagen.flac import FLAC, Picture
from winsdk.windows.media.control import (
    GlobalSystemMediaTransportControlsSessionManager as SessionManager,
)
from winsdk.windows.storage.streams import DataReader


# Configuration Constants
SAMPLE_RATE = 48_000
CHANNELS = 2
OUTPUT_DIRECTORY = Path("output")
METADATA_POLL_SECONDS = 0.1

# Time in seconds before the metadata API update that the new track actually started
SONG_SPLIT_OFFSET_SECONDS = 0.06

stop_event = threading.Event()

# Sentinel object for queue timeouts
TIMEOUT_SENTINEL = object()

audio_queue: queue.Queue[np.ndarray | object | None] = queue.Queue()
metadata_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()


@dataclass
class SongMetadata:
    title: str = ""
    artist: str = ""
    album: str = ""
    album_artist: str = ""
    track_number: int = 0
    album_track_count: int = 0
    genre: str = ""
    playback_type: str = ""
    source_app: str = ""
    started_at: str = ""
    album_art: bytes | None = None

    def key(self) -> tuple:
        return (
            self.title,
            self.artist,
            self.album,
            self.album_artist,
            self.track_number,
            self.source_app,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "artist": self.artist,
            "album": self.album,
            "album_artist": self.album_artist,
            "track_number": self.track_number,
            "album_track_count": self.album_track_count,
            "genre": self.genre,
            "playback_type": self.playback_type,
            "source_app": self.source_app,
            "started_at": self.started_at,
            "album_art": self.album_art,
        }


def sanitize_filename(value: str, fallback: str = "unknown_song") -> str:
    value = value.strip() or fallback
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
    value = value.rstrip(". ")
    return value[:180] or fallback


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    counter = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


async def get_album_art(session: Any) -> bytes | None:
    try:
        properties = await session.try_get_media_properties_async()
        thumbnail = properties.thumbnail
        if thumbnail is None:
            return None

        random_access_stream = await thumbnail.open_read_async()
        input_stream = random_access_stream.get_input_stream_at(0)
        reader = DataReader(input_stream)

        try:
            size = int(random_access_stream.size)
            if size <= 0:
                return None

            await reader.load_async(size)
            buffer = reader.read_buffer(size)
            if buffer is None:
                return None

            return bytes(cast(Any, buffer))
        finally:
            reader.close()
            input_stream.close()
            random_access_stream.close()
    except Exception:
        return None


async def get_windows_media_metadata() -> SongMetadata | None:
    manager = await SessionManager.request_async()
    session = manager.get_current_session()

    if session is None:
        return None

    properties = await session.try_get_media_properties_async()
    album_art = await get_album_art(session)

    source_app = ""
    try:
        source_app = session.source_app_user_model_id
    except AttributeError:
        pass

    return SongMetadata(
        title=properties.title or "",
        artist=properties.artist or "",
        album=properties.album_title or "",
        album_artist=properties.album_artist or "",
        track_number=properties.track_number,
        album_track_count=properties.album_track_count,
        genre=properties.genres[0] if properties.genres else "",
        source_app=source_app,
        started_at=datetime.now(timezone.utc).isoformat(),
        album_art=album_art,
    )


def metadata_thread():
    previous_key: tuple | None = None

    async def monitor():
        nonlocal previous_key

        while not stop_event.is_set():
            try:
                metadata = await get_windows_media_metadata()

                if metadata is not None:
                    current_key = metadata.key()

                    if current_key != previous_key and metadata.title:
                        metadata_queue.put(metadata.to_dict())
                        previous_key = current_key
                        print(f"Media change: {metadata.artist} - {metadata.title}")

            except Exception as exc:
                print("Metadata error:", exc)

            await asyncio.sleep(METADATA_POLL_SECONDS)

    asyncio.run(monitor())


def recording_thread(device_info: dict[str, Any]):
    p = pyaudio.PyAudio()

    def callback(in_data, frame_count, time_info, status):
        if stop_event.is_set():
            return (None, pyaudio.paComplete)

        audio_data = np.frombuffer(in_data, dtype=np.int16)
        audio_data = audio_data.reshape(-1, CHANNELS)
        audio_queue.put(audio_data.copy())

        return (None, pyaudio.paContinue)

    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=device_info["index"],
            stream_callback=callback,
        )

        print("Recording...")
        stream.start_stream()

        while stream.is_active() and not stop_event.is_set():
            stop_event.wait(0.1)

        stream.stop_stream()
        stream.close()

    except Exception as exc:
        print("Recording error:", exc)
        stop_event.set()

    finally:
        p.terminate()
        audio_queue.put(None)


def detect_image_mime_type(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def save_song(
    chunks: list[np.ndarray],
    metadata: dict[str, Any],
):
    if not chunks:
        return

    audio = np.concatenate(chunks, axis=0)

    title = metadata.get("title") or "unknown_song"
    artist = metadata.get("artist") or "unknown_artist"
    album = metadata.get("album") or ""
    album_artist = metadata.get("album_artist") or ""
    genre = metadata.get("genre") or ""
    track_number = metadata.get("track_number") or 0
    album_art = metadata.get("album_art")

    base_name = sanitize_filename(f"{artist} - {title}")
    flac_path = unique_path(OUTPUT_DIRECTORY / f"{base_name}.flac")

    # 1. Write PCM audio to FLAC container
    sf.write(str(flac_path), audio, SAMPLE_RATE, format="FLAC", subtype="PCM_16")

    # 2. Attach Vorbis Comments and Album Art
    flac = FLAC(str(flac_path))

    flac["title"] = title
    flac["artist"] = artist
    if album:
        flac["album"] = album
    if album_artist:
        flac["albumartist"] = album_artist
    if genre:
        flac["genre"] = genre
    if track_number:
        flac["tracknumber"] = str(track_number)

    if album_art:
        picture = Picture()
        picture.type = 3  # Front Cover
        picture.mime = detect_image_mime_type(album_art)
        picture.desc = "Cover"
        picture.data = album_art
        flac.clear_pictures()
        flac.add_picture(picture)

    flac.save()

    print(f"Saved: {flac_path}")


def saving_thread():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    current_metadata: dict[str, Any] | None = None
    current_chunks: list[tuple[float, np.ndarray]] = []

    while True:
        try:
            chunk = audio_queue.get(timeout=0.1)
        except queue.Empty:
            chunk = TIMEOUT_SENTINEL

        now = datetime.now(timezone.utc).timestamp()

        if not metadata_queue.empty():
            new_metadata = metadata_queue.get()

            if current_metadata is not None and current_chunks:
                split_threshold = now - SONG_SPLIT_OFFSET_SECONDS
                old_song_chunks: list[np.ndarray] = [c for t, c in current_chunks if t < split_threshold]
                next_song_chunks: list[tuple[float, np.ndarray]] = [
                    (t, c) for t, c in current_chunks if t >= split_threshold
                ]

                save_song(old_song_chunks, current_metadata)
                current_chunks = next_song_chunks

            current_metadata = new_metadata

        if chunk is None:
            break

        if chunk is not TIMEOUT_SENTINEL and isinstance(chunk, np.ndarray):
            if current_metadata is None:
                current_metadata = {
                    "title": "unknown_song",
                    "artist": "unknown_artist",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                }
            current_chunks.append((now, chunk))

    if current_metadata is not None and current_chunks:
        save_song([c for _, c in current_chunks], current_metadata)

    print("Saving complete")


def find_wasapi_loopback_device() -> dict[str, Any]:
    p = pyaudio.PyAudio()

    try:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    except OSError:
        p.terminate()
        raise RuntimeError("WASAPI host API not available on this system.")

    default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

    if not default_speakers.get("isLoopbackDevice"):
        for loopback in p.get_loopback_device_info_generator():
            if default_speakers["name"] in loopback["name"]:
                p.terminate()
                return loopback

    p.terminate()
    raise RuntimeError("Could not find a WASAPI loopback device.")


def main(device_info: dict[str, Any]):
    print("Welcome to albac!")
    print(f"Recording device ID: {device_info['index']}")
    print(f"Recording device: {device_info['name']}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Channels: {CHANNELS}")
    print(f"Output directory: {OUTPUT_DIRECTORY.resolve()}")
    print(f"Song split offset: {SONG_SPLIT_OFFSET_SECONDS} seconds")

    recorder = threading.Thread(
        target=recording_thread,
        args=(device_info,),
        name="Recorder",
    )
    saver = threading.Thread(
        target=saving_thread,
        name="Saver",
    )
    metadata = threading.Thread(
        target=metadata_thread,
        name="Metadata",
    )

    saver.start()
    metadata.start()
    recorder.start()

    try:
        input("Press Enter to stop recording...\n")
    finally:
        stop_event.set()

    recorder.join()
    metadata.join()
    saver.join()


if __name__ == "__main__":
    device_info = find_wasapi_loopback_device()
    main(device_info)