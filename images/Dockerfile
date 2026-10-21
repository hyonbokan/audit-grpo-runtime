# Pinned runtime images, rebuilt for CUDA 12.9 from the frozen recipe in
# workspace/configs/dependencies.Dockerfile. Only CUDA wheels differ from the original locks.
#   deps:  /opt/pilot/judge (Python 3.14, judge) and /opt/pilot/gpu (Python 3.12, native update)
#   vllm:  the same judge environment; /opt/pilot/gpu also holds vLLM 0.31.0+cu129 (rollout)
# Build with images/build.sh: it fills a wheel directory with images/fetch_wheels.py and
# passes it as the `wheels` build context, so every install below runs offline.
ARG BASE_IMAGE=ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55
FROM ghcr.io/astral-sh/uv:0.12.7@sha256:95f2aa1fe59274951cfe9b0cbc7972e879ff1004bc8945d130a32eb0dbd85945 AS uv
# Replaced by `--build-context wheels=DIR`; empty otherwise, which fails the installs.
FROM scratch AS wheels

FROM ${BASE_IMAGE} AS base
COPY --from=uv /uv /uvx /usr/local/bin/
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates build-essential git libgomp1 pkg-config && \
    rm -rf /var/lib/apt/lists/*
ENV UV_PYTHON_INSTALL_DIR=/opt/pilot/python \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    UV_INDEX_STRATEGY=unsafe-best-match \
    HF_HOME=/workspace/cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PATH=/opt/pilot/gpu/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
COPY dependency_check.py /opt/pilot/dependency_check.py
COPY judge-requirements.lock /opt/pilot/locks/
RUN --mount=type=bind,from=wheels,target=/wheels \
    uv venv --python cpython-3.14.7-linux-x86_64-gnu /opt/pilot/judge && \
    uv pip sync --offline --no-index --find-links /wheels --python /opt/pilot/judge/bin/python --require-hashes --only-binary :all: \
    /opt/pilot/locks/judge-requirements.lock && \
    /opt/pilot/judge/bin/python /opt/pilot/dependency_check.py /opt/pilot/locks/judge-requirements.lock
ENV TIKTOKEN_CACHE_DIR=/opt/pilot/tokenizer-cache
RUN /opt/pilot/judge/bin/python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"
WORKDIR /workspace/code/backend
CMD ["/bin/bash"]

FROM base AS deps
COPY gpu-requirements-cu129.lock /opt/pilot/locks/
RUN --mount=type=bind,from=wheels,target=/wheels \
    uv venv --python cpython-3.12.14-linux-x86_64-gnu /opt/pilot/gpu && \
    uv pip sync --offline --no-index --find-links /wheels --python /opt/pilot/gpu/bin/python --require-hashes --only-binary :all: \
    /opt/pilot/locks/gpu-requirements-cu129.lock && \
    uv pip check --python /opt/pilot/gpu/bin/python && \
    /opt/pilot/gpu/bin/python /opt/pilot/dependency_check.py /opt/pilot/locks/gpu-requirements-cu129.lock
RUN /opt/pilot/gpu/bin/python -c \
    "import torch; assert torch.version.cuda == '12.9', torch.version.cuda; print(torch.__version__)"

FROM base AS vllm
COPY vllm-requirements-cu129.lock /opt/pilot/locks/
RUN --mount=type=bind,from=wheels,target=/wheels \
    uv venv --python cpython-3.12.14-linux-x86_64-gnu /opt/pilot/gpu && \
    uv pip sync --offline --no-index --find-links /wheels --python /opt/pilot/gpu/bin/python --require-hashes --only-binary :all: \
    /opt/pilot/locks/vllm-requirements-cu129.lock && \
    uv pip check --python /opt/pilot/gpu/bin/python && \
    /opt/pilot/gpu/bin/python /opt/pilot/dependency_check.py /opt/pilot/locks/vllm-requirements-cu129.lock
RUN /opt/pilot/gpu/bin/python -c \
    "import torch; assert torch.version.cuda == '12.9', torch.version.cuda; print(torch.__version__)"
