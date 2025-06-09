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
    "whisper-base",
    "alphacep/vosk-model-ru",
    "alphacep/vosk-model-small-ru",
    "onnx-community/whisper-tiny",
    "onnx-community/whisper-base",
    "onnx-community/whisper-small",
    "onnx-community/whisper-large-v3-turbo"
]

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

    onnx_asr_version = "0.6.1"  # Фиксированная версия из pip show onnx-asr

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
                        languages=["ru", "en"],
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