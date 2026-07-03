PYTHON ?= python3

.PHONY: install test eval demo

install:
	$(PYTHON) -m pip install -r bug_classifier/requirements.txt

test:
	NOTION_DRY_RUN=true $(PYTHON) -m pytest bug_classifier/tests -q

eval:
	NOTION_DRY_RUN=true $(PYTHON) -m pytest bug_classifier/tests/test_evaluation.py -q -s

demo:
	$(PYTHON) -m bug_classifier.main --demo --dry-run
