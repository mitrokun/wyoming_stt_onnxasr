# Wyoming STT сервер для Home Assistant на базе [OnnxAsr](https://github.com/istupakov/onnx-asr)
### Нацелен на русскоговорящую аудиторию.

```
# В простом случае достаточно установить зависимости, скачать каталог с сервером и запустить его
# Детали смотрите в репозитории OnnxAsr
pip install onnx-asr[cpu,hub] wyoming

# Пример для win с кастомным портом и облегченной версией модели (int8)
python -m wyoming_onnxasr --model gigaam-v2-ctc --uri 'tcp://0.0.0.0:10305' --quantization int8

# Если установлен onnxruntime-gpu, то используйте --device cuda
# В linux, как водится, все операции выполняйте в виртуальной среде.
```

### Доступные модели:
```
gigaam-v2-ctc    # это база, int8 - 240 мб, full - 900мб
gigaam-v2-rnnt
nemo-fastconformer-ru-ctc
nemo-fastconformer-ru-rnnt
nemo-parakeet-ctc-0.6b
nemo-parakeet-rnnt-0.6b
nemo-parakeet-tdt-0.6b-v2
whisper-base
alphacep/vosk-model-ru    # int - 70мб, full - 260мб, ещё быстрее но WER хуже
alphacep/vosk-model-small-ru
onnx-community/whisper-tiny
onnx-community/whisper-base
onnx-community/whisper-small
onnx-community/whisper-large-v3-turbo
```
