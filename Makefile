.PHONY: data ingest features train dashboard drift test lint docker all

data:
	python3 -m src.ingestion.generate_synthetic_data

ingest:
	python3 -m src.ingestion.load_raw

features:
	python3 -m src.features.setup_feature_store

train:
	python3 -m src.training.train

drift:
	python3 -m src.monitoring.drift_report

simulate:
	python3 -m src.monitoring.simulate_new_batch

dashboard:
	streamlit run dashboard/app.py

test:
	pytest tests/ -v

lint:
	ruff check src/

docker:
	docker build -f docker/Dockerfile -t pulseml .
	docker run -p 8501:8501 pulseml

all: data ingest features train drift
