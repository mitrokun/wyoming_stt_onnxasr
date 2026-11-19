FROM python:3.13-slim

ARG WORK_UID="1000"
ARG WORK_GID="1000"
ARG WORK_USER="wonnx"
ARG WORK_GROUP="wonnx"
ARG WORK_HOME="/app"
ARG WORK_SHELL="/bin/bash"

ENV HOME="/app"

LABEL org.mitrokun.title="mitrokun/wyoming-onnxasr" \
    org.mitrokun.version="1.0"

RUN groupadd -g "${WORK_GID}" "${WORK_GROUP}" \
    && useradd -m -d "${WORK_HOME}" -s "${WORK_SHELL}" -u "${WORK_UID}" -g "${WORK_GID}" "${WORK_USER}"

COPY requirements.txt ${WORK_HOME}/requirements.txt

RUN pip install -U pip --no-cache \
    && pip install --no-cache -r ${WORK_HOME}/requirements.txt

COPY wyoming_onnxasr/ ${WORK_HOME}/wyoming_onnxasr/
COPY script/ ${WORK_HOME}/script/

RUN chmod ugo+x ${WORK_HOME}/script/run-docker \
    && mkdir -p ${WORK_HOME}/.cache/huggingface \
    && chown -R "${WORK_USER}":"${WORK_GROUP}" "${WORK_HOME}"

WORKDIR "${WORK_HOME}"
USER "${WORK_USER}":"${WORK_GROUP}"

ENTRYPOINT ["/app/script/run-docker"]
