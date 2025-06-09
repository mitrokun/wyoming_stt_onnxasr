"""Event handler for clients of the server."""
import argparse
import asyncio
import logging
from typing import Any

import numpy as np
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop, AudioStart
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)

class OnnxAsrEventHandler(AsyncEventHandler):
    """Event handler for clients using ONNX ASR models."""

    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        model: Any,
        model_lock: asyncio.Lock,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.model = model
        self.model_lock = model_lock
        self.audio_buffer: bytearray | None = None
        self.sample_rate = 16000
        self.sample_width = 2
        self.channels = 1

    async def handle_event(self, event: Event) -> bool:
        if AudioStart.is_type(event.type):
            audio_start = AudioStart.from_event(event)
            self.sample_rate = audio_start.rate
            self.sample_width = audio_start.width
            self.channels = audio_start.channels
            self.audio_buffer = bytearray()
            _LOGGER.debug(f"Audio started: {self.sample_rate} Hz, {self.sample_width*8}-bit, {self.channels} channel(s)")
            return True

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)

            if self.audio_buffer is None:
                self.audio_buffer = bytearray()

            self.audio_buffer.extend(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug("Audio stopped. Transcribing...")
            assert self.audio_buffer is not None

            audio_s16 = np.frombuffer(self.audio_buffer, dtype=np.int16)
            audio_f32 = audio_s16.astype(np.float32) / 32768.0

            async with self.model_lock:
                transcription = self.model.recognize(audio_f32, sample_rate=self.sample_rate)

            self.audio_buffer = None

            _LOGGER.info(f"Transcription: {transcription}")

            await self.write_event(Transcript(text=transcription).event())
            _LOGGER.debug("Completed request")

            return True

        if Transcribe.is_type(event.type):
            self.audio_buffer = None
            return True

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        return True