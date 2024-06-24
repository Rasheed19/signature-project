setup:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

create-required-dir:
	mkdir plots
	mkdir data