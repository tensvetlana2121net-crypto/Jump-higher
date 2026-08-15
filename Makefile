.PHONY: install lint test run worker bot migrate

install:
	pip install -e ".[dev,cv]"

lint:
	ruff check .
	mypy src

test:
	pytest

run:
	uvicorn jumpbot.api.main:app --reload

worker:
	celery -A jumpbot.worker.celery_app worker -l INFO

bot:
	python -m jumpbot.bot.main

migrate:
	alembic upgrade head
