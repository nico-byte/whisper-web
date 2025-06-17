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
	source .env && SERVER_TYPE=cpu docker compose --profile cpu --profile streamlit up

docker.up.server-cuda-streamlit:
	source .env && SERVER_TYPE=cuda docker compose --profile cuda --profile streamlit up

docker.build.server-cpu:
	source .env && docker compose build server-cpu

docker.build.server-cuda:
	source .env && docker compose build server-cuda

docker.build.streamlit:
	source .env && docker compose build streamlit-upload
	source .env && docker compose build streamlit-viewer

docker.build.all:
	source .env && docker compose build server-cpu
	source .env && docker compose build server-cuda
	source .env && docker compose build streamlit-upload
	source .env && docker compose build streamlit-viewer

docker.clean:
	source .env && docker compose down -v --remove-orphans
	source .env && docker system prune -f --volumes
	source .env && docker image prune -f

# Local development tasks - using the local venv
local.server:
	source .env && source ./.venv/bin/activate && ./.venv/bin/python -m app.server

local.cli:
	source .env && source ./.venv/bin/activate && ./.venv/bin/python -m app.cli

local.viewer:
	source .env && source ./.venv/bin/activate && ./.venv/bin/python -m streamlit run ./app/streamlit_viewer_client.py --server.address 127.0.0.1

local.upload:
	source .env && source ./.venv/bin/activate && ./.venv/bin/python -m streamlit run ./app/streamlit_upload_client.py --server.address 127.0.0.1

local.run.cli:
	@echo "Starting server and cli..."
	@trap 'echo "Stopping..."; kill 0' SIGINT SIGTERM EXIT; \
	( \
		source .env && source ./.venv/bin/activate && ./.venv/bin/python -m app.server & \
		echo "Server started" & \
		sleep 10 && \
		source .env && source ./.venv/bin/activate && ./.venv/bin/python -m app.cli & \
		echo "Cli started" & \
		source .env && source ./.venv/bin/activate && ./.venv/bin/python -m streamlit run ./app/streamlit_viewer_client.py --server.address 127.0.0.1 & \
		echo "Streamlit app started" & \
		wait \
	)

local.run.upload:
	@echo "Starting server and upload frontend..."
	@trap 'echo "Stopping..."; kill 0' SIGINT SIGTERM EXIT; \
	( \
		source .env && source ./.venv/bin/activate && ./.venv/bin/python -m app.server & \
		echo "Server started" & \
		sleep 10 && \
		source .env && source ./.venv/bin/activate && ./.venv/bin/python -m streamlit run ./app/streamlit_upload_client.py --server.address 127.0.0.1 & \
		echo "Upload frontend started" & \
		wait \
	)