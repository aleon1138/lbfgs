format:
	ruff format lbfgs/*.py test/*.py bench/*.py

lint:
	ruff check lbfgs/*.py test/*.py bench/*.py

test:
	python -m pytest test/ -v

profile:
	python bench/profile_lbfgs.py

.PHONY: format lint test profile
