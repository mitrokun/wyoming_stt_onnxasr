## Wyoming STT сервер для Home Assistant на базе [ONNX ASR](https://github.com/istupakov/onnx-asr)
#### Нацелен на русскоговорящую аудиторию (из многообразия моделей рекомендую `gigaam-v3`)
For English-speaking users, the Parakeet models (e.g., `nemo-parakeet-tdt-0.6b-v2`) are recommended, `canary-1b-v2` is slightly higher quality, but 4 times slower.

```
# В простом случае достаточно установить зависимости, скачать каталог с сервером и запустить его
# Детали смотрите в репозитории OnnxAsr
pip install onnx-asr[cpu,hub] wyoming

# Пример для win с кастомным портом и облегченной версией модели (int8)
python -m wyoming_onnxasr --model gigaam-v3-ctc --uri 'tcp://0.0.0.0:10305' --quantization int8

# Если установлен onnxruntime-gpu, то используйте --device cuda
pip install onnxruntime-gpu
```
В linux, как водится, все операции выполняйте в виртуальной среде вручную. Или воспользуйтесь скриптами:
```
git clone https://github.com/mitrokun/wyoming_stt_onnxasr.git
cd wyoming_stt_onnxasr
script/setup
# script/run запустит сервер с параметрами из примера выше, но с полной моделью, используйте ключи для конфигурации
# можно доустановить библиотеку для cuda
~/wyoming_stt_onnxasr/.venv/bin/pip install onnxruntime-gpu
```

### Доступные модели:
```
gigaam-v2-ctc
gigaam-v2-rnnt
gigaam-v3-ctc                 # теперь это база, int8 - 220 мб, full - 865мб
gigaam-v3-rnnt
gigaam-v3-e2e-ctc             # e2e модели с пунктуацией
gigaam-v3-e2e-rnnt
alphacep/vosk-model-ru        # int8 - 70мб, full - 260мб, ещё быстрее но WER хуже
alphacep/vosk-model-small-ru  # 25мб/90мб, в аддоне HA исользуется v0.22, здесь v0.54
nemo-fastconformer-ru-ctc
nemo-fastconformer-ru-rnnt
nemo-parakeet-ctc-0.6b        # eng / int8 - 620мб
nemo-parakeet-rnnt-0.6b       # eng
nemo-parakeet-tdt-0.6b-v2     # eng
nemo-parakeet-tdt-0.6b-v3     # multilingual / may work inconsistently, as automatic language detection is used for each request
nemo-canary-1b-v2             # multilingual / int8 - 980мб; The transcription language is matched with the agent language.

whisper-base                  # комплектные варианты шёпота не интересны, так как есть faster-whisper
onnx-community/whisper-tiny
onnx-community/whisper-base
onnx-community/whisper-small
onnx-community/whisper-large-v3-turbo
```
`parakeet-v3` и `canary-1b-v2` поддерживают следующие языки: `"bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk"`

### Aддон для HA (только cpu):
[![Open your Home Assistant instance and show the add-on store for this repository.](https://my.home-assistant.io/badges/supervisor_addon.svg)](https://my.home-assistant.io/redirect/supervisor_addon/?addon=f4ba1342_onnxasr&repository_url=https%3A%2F%2Fgithub.com%2Fmitrokun%2Fvoice-addons)

### Запуск в Docker

#### Локальная сборка образа
Клонируйте репозиторий и выполните сборку образа из корневой директории проекта:
```bash
docker build -t mitrokun/wyoming-onnxasr:latest .
```

#### Управление через Docker Compose
Запуск контейнера в фоновом режиме:
```bash
docker compose up -d
```

Просмотр логов работы сервера:
```bash
docker compose logs -f
```

Остановка и удаление контейнера (сохраненный том с моделью при этом не удаляется):
```bash
docker compose down
```

#### Особенности конфигурации и хранения моделей
* **Параметры запуска:** Аргументы в блоке `command` (`--model` и `--uri`) являются обязательными. Остальные флаги (например, `--quantization int8`) опциональны и полностью совпадают с параметрами ручного запуска приложения.
* **Кэширование моделей:** Выбранная модель загружается один раз при первом старте контейнера и сохраняется в именованном томе Docker, поэтому при последующих перезапусках повторное скачивание не требуется. При необходимости в файле `docker-compose.yml` вместо именованного тома можно примонтировать собственную директорию с хоста (например, `/path/to/local/folder:/app/.cache/huggingface`).
