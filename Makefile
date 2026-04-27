PYTHON = python
PYTHONPATH = src

.PHONY: collect features train predict run update plots clean

collect:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m src collect

features:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m src features

train:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m src train

predict:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m src predict

run:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m src run

update:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m src update

plots:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m src plots

clean:
	rm -f data/features.csv data/cv_results.csv data/predictions_2026*.csv
	rm -f models/*.pkl
