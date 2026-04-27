PYTHON = python

.PHONY: collect features train predict run update plots clean

collect:
	$(PYTHON) -m grid_prophet collect

features:
	$(PYTHON) -m grid_prophet features

train:
	$(PYTHON) -m grid_prophet train

predict:
	$(PYTHON) -m grid_prophet predict

run:
	$(PYTHON) -m grid_prophet run

update:
	$(PYTHON) -m grid_prophet update

plots:
	$(PYTHON) -m grid_prophet plots

clean:
	rm -f data/features.csv data/cv_results.csv data/predictions_2026*.csv
	rm -f models/*.pkl
