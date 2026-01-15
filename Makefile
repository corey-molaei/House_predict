.PHONY: ingest features train evaluate

ingest:
	python -m src.ingest.kaggle_ingest
	python -m src.ingest.domain_api_ingest
	python -m src.ingest.nsw_ingest

features:
	python -m src.features.build_features

train:
	python -m src.models.train

evaluate:
	python -m src.models.evaluate
