.PHONY: help install dev-install format lint test clean docker-build docker-up docker-down

help:
	@echo "AdverScan Management Commands"
	@echo "-----------------------------"
	@echo "make install      : Install production dependencies"
	@echo "make dev-install  : Install development dependencies in editable mode"
	@echo "make format       : Format code using Black and isort"
	@echo "make lint         : Run code linting (flake8, mypy)"
	@echo "make test         : Run tests using pytest"
	@echo "make clean        : Remove temporary files and cache"
	@echo "make docker-build : Build Docker container"
	@echo "make docker-up    : Start services via docker-compose"
	@echo "make docker-down  : Stop docker-compose services"

install:
	pip install -r requirements.txt

dev-install:
	pip install -e .[dev]

format:
	black app/ tests/
	isort app/ tests/

lint:
	flake8 app/ tests/
	mypy app/

test:
	pytest tests/ --cov=app

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov *.egg-info build dist

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down
