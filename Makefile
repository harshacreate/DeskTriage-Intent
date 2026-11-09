.PHONY: setup train test
setup:
	python3 -m pip install -e '.[test]'
train:
	python3 -m desktriage.train
test:
	python3 -m pytest -q
