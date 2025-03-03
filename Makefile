# Makefile for YouTube Frame Extractor
# -----------------------------------
# Available targets:
#   help      - Display this help message
#   install   - Install Python dependencies
#   test      - Run tests with pytest
#   lint      - Run code linting and type checks (black, isort, flake8, mypy)
#   format    - Format code with black and isort
#   docs      - Build documentation using mkdocs
#   run       - Run the YouTube Frame Extractor CLI tool
#   docker    - Build and run the Docker container
#   clean     - Remove temporary files and build artifacts

.PHONY: help install test lint format docs run docker clean

help:
	@echo "Makefile commands for YouTube Frame Extractor:"
	@echo "  make install   - Install Python dependencies"
	@echo "  make test      - Run tests with pytest"
	@echo "  make lint      - Run linting (black, isort, flake8, mypy)"
	@echo "  make format    - Format code with black and isort"
	@echo "  make docs      - Build documentation with mkdocs"
	@echo "  make run       - Run the CLI tool (python -m youtube_frame_extractor)"
	@echo "  make docker    - Build and run the Docker container"
	@echo "  make clean     - Remove temporary files and build artifacts"

install:
	@echo "Installing dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install git+https://github.com/openai/CLIP.git

test:
	@echo "Running tests..."
	pytest --maxfail=1 --disable-warnings -q

lint:
	@echo "Running linters and type checks..."
	black --check .
	isort --check-only .
	flake8 .
	mypy .

format:
	@echo "Formatting code..."
	black .
	isort .

docs:
	@echo "Building documentation..."
	mkdocs build

run:
	@echo "Running YouTube Frame Extractor CLI..."
	python -m youtube_frame_extractor

docker:
	@echo "Building Docker image..."
	docker build -t youtube-frame-extractor .
	@echo "Running Docker container..."
	docker run --rm -it youtube-frame-extractor

clean:
	@echo "Cleaning up temporary files and build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build dist
