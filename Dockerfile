FROM python:3.10-slim
ENV PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_HOME=/opt/poetry \
    VENV_PATH=/opt/venv \
    POETRY_VERSION=1.8.3
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH:/root/.local/bin"

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install  -y \
        # deps for installing poetry
        curl \
        build-essential \
        ffmpeg libsm6 libxext6 \
    \
    && curl -sSL curl -sSL https://install.python-poetry.org | POETRY_HOME=$POETRY_HOME python3 - \
    && poetry --version \
    \
    # configure poetry & make a virtualenv ahead of time since we only need one
    && python -m venv $VENV_PATH \
    && poetry config virtualenvs.create false \
    \
    # cleanup
    && rm -rf /var/lib/apt/lists/*

COPY poetry.lock pyproject.toml ./
RUN poetry install --no-interaction --no-ansi -vvv

WORKDIR /app

COPY . ./

RUN ./download_models.sh

EXPOSE 8000

ENTRYPOINT ["poetry", "run", "uvicorn", "app_v2:app", "--port", "8000", "--host", "0.0.0.0"]
