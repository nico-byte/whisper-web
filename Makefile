.PHONY: format lint lintfix pytest
.PHONY: docker..up.server-cpu-streamlit docker.up.server-cuda-streamlit docker.build.server-cpu 
.PHONY: docker.build.server-cuda docker.build.streamlit docker.build.all docker.clean
.PHONY: local.server local.cli local.upload local.viewer local.run.cli local.run.upload

SHELL := /bin/bash

# Development tasks
format:
	uv run ruff format

lint:
	uv run ruff check

lintfix:
	uv run ruff check --fix

pytest:
	source ./.venv/bin/activate && ./.venv/bin/python -m pytest ./tests -v --tb=short

# Docker tasks
docker.up.server-cpu-streamlit:
	SERVER_TYPE=cpu docker compose --profile cpu --profile streamlit up

docker.up.server-cuda-streamlit:
	SERVER_TYPE=cuda docker compose --profile cuda --profile streamlit up

docker.build.server-cpu:
	docker compose build server-cpu

docker.build.server-cuda:
	docker compose build server-cuda

docker.build.streamlit:
	docker compose build streamlit-upload
	docker compose build streamlit-viewer

docker.build.all-cpu:
	docker compose build server-cpu
	docker compose build streamlit-upload
	docker compose build streamlit-viewer

docker.build.all-cuda:
	docker compose build server-cuda
	docker compose build streamlit-upload
	docker compose build streamlit-viewer

# Local development tasks - using the local venv
local.server:
	uv sync --group server --group client --extra cpu --python python3.12 && source ./.venv/bin/activate && ./.venv/bin/python -m app.server

local.cli:
	source ./.venv/bin/activate && ./.venv/bin/python -m app.cli

local.viewer:
	source ./.venv/bin/activate && ./.venv/bin/python -m streamlit run ./app/streamlit_viewer_client.py --server.address 127.0.0.1

local.upload:
	uv sync --group streamlit --group client --group http --extra cpu --python python3.12 && source ./.venv/bin/activate && ./.venv/bin/python -m streamlit run ./app/streamlit_upload_client.py --server.address 127.0.0.1

local.diarization:
	uv sync --group streamlit --group client --group http --extra cpu --python python3.12 && source ./.venv/bin/activate && ./.venv/bin/python -m streamlit run ./app/streamlit_diarization.py --server.address 127.0.0.1

local.run.cli:
	@echo "Starting server and cli..."
	@trap 'echo "Stopping..."; kill 0' SIGINT SIGTERM EXIT; \
	( \
		source ./.venv/bin/activate && ./.venv/bin/python -m app.server & \
		echo "Server started" & \
		sleep 10 && \
		source ./.venv/bin/activate && ./.venv/bin/python -m app.cli & \
		echo "Cli started" & \
		source ./.venv/bin/activate && ./.venv/bin/python -m streamlit run ./app/streamlit_viewer_client.py --server.address 127.0.0.1 & \
		echo "Streamlit app started" & \
		wait \
	)

local.run.upload:
	@echo "Starting server and upload frontend..."
	@trap 'echo "Stopping..."; kill 0' SIGINT SIGTERM EXIT; \
	( \
		source ./.venv/bin/activate && ./.venv/bin/python -m app.server & \
		echo "Server started" & \
		sleep 10 && \
		source ./.venv/bin/activate && ./.venv/bin/python -m streamlit run ./app/streamlit_upload_client.py --server.address 127.0.0.1 & \
		echo "Upload frontend started" & \
		wait \
	)