#!/usr/bin/env python3
import argparse
import asyncio
import logging
from functools import partial

import onnx_asr
import numpy as np
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from . import __version__
from .handler import OnnxAsrEventHandler

_LOGGER = logging.getLogger(__name__)

SUPPORTED_MODELS = [
    "gigaam-v2-ctc",
    "gigaam-v2-rnnt",
    "nemo-fastconformer-ru-ctc",
    "nemo-fastconformer-ru-rnnt",
    "nemo-parakeet-ctc-0.6b",
    "nemo-parakeet-rnnt-0.6b",
    "nemo-parakeet-tdt-0.6b-v2",
    "nemo-parakeet-tdt-0.6b-v3",
    "whisper-base",
    "alphacep/vosk-model-ru",
    "alphacep/vosk-model-small-ru",
    "onnx-community/whisper-tiny",
    "onnx-community/whisper-base",
    "onnx-community/whisper-small",
    "onnx-community/whisper-large-v3-turbo"
]

DEFAULT_LANGUAGES = ["ru", "en"]

WHISPER_LANGUAGES = [
    "bg", "ca", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "he",
    "hi", "hr", "hu", "id", "is", "it", "ja", "ko", "lt", "lv", "my", "nl",
    "no", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "sw", "ta", "th", "tr",
    "uk", "ur", "vi", "zh" 
]

MODEL_LANGUAGES = {
    # Multilanguage
    "nemo-parakeet-tdt-0.6b-v3": [
        "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu",
        "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk"
    ],
    # EN
    "nemo-parakeet-ctc-0.6b": ["en"],
    "nemo-parakeet-rnnt-0.6b": ["en"],
    "nemo-parakeet-tdt-0.6b-v2": ["en"],
    # RU
    "gigaam-v2-ctc": ["ru"],
    "gigaam-v2-rnnt": ["ru"],
    "alphacep/vosk-model-ru": ["ru"],
    "alphacep/vosk-model-small-ru": ["ru"],
    "nemo-fastconformer-ru-ctc": ["ru"],
    "nemo-fastconformer-ru-rnnt": ["ru"],
    # Whisper
    "whisper-base": WHISPER_LANGUAGES,
    "onnx-community/whisper-tiny": WHISPER_LANGUAGES,
    "onnx-community/whisper-base": WHISPER_LANGUAGES,
    "onnx-community/whisper-small": WHISPER_LANGUAGES,
    "onnx-community/whisper-large-v3-turbo": WHISPER_LANGUAGES,
}


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help=f"Name of the model to load from HuggingFace (e.g., gigaam-v2-ctc). Supported models: {', '.join(SUPPORTED_MODELS)}",
    )
    parser.add_argument(
        "--quantization",
        choices=["int8", "fp16", "fp32"],
        default=None,
        help="Model quantization to use (if available). Default is fp32.",
    )
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use for inference ('cpu' or 'cuda'). Requires correct onnxruntime installation.",
    )
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    parser.add_argument(
        "--log-format", default=logging.BASIC_FORMAT, help="Format for log messages"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print version and exit",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format
    )
    _LOGGER.debug(args)

    onnx_asr_version = "0.7.0"

    supported_languages = MODEL_LANGUAGES.get(args.model, DEFAULT_LANGUAGES)

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="onnx-asr",
                description="Speech-to-text with ONNX Runtime",
                attribution=Attribution(
                    name="istupakov",
                    url="https://github.com/istupakov/onnx-asr",
                ),
                installed=True,
                version=onnx_asr_version,
                models=[
                    AsrModel(
                        name=args.model,
                        description=f"ONNX model: {args.model}",
                        attribution=Attribution(
                            name="HuggingFace Community",
                            url=f"https://huggingface.co/istupakov/{args.model}",
                        ),
                        installed=True,
                        languages=supported_languages,
                        version=args.quantization or "fp32",
                    )
                ],
            )
        ],
    )

    _LOGGER.info(
        f"Loading model '{args.model}' (quantization: {args.quantization or 'fp32'}) on device '{args.device}'"
    )

    providers = []
    if args.device == "cuda":
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    model = onnx_asr.load_model(
        args.model,
        quantization=args.quantization,
        providers=providers
    )
    _LOGGER.info(f"Model '{args.model}' loaded successfully.")

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready")
    model_lock = asyncio.Lock()
    await server.run(
        partial(
            OnnxAsrEventHandler,
            wyoming_info,
            args,
            model,
            model_lock,
        )
    )


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
