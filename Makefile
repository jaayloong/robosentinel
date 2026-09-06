install:
	pip install -e ".[dev]"

test:
	pytest -q

run:
	uvicorn robosentinel.api:app --reload

demo-data:
	robosentinel simulate --rows 1500 --seed 42 --output telemetry_demo.csv
