.DEFAULT_GOAL := lib
.PHONY: help lib examples check check-extended check-asan check-tsan check-plan check-generated verify-donors verify-evidence audit bench
PYTHON ?= python3
help:
	@echo "Clang-only native library: make lib / examples / check / check-extended / audit"
	@echo "Experiments are optional: clone libsbn_experiments into experiments/."
lib:
	$(PYTHON) tools/build.py lib
examples:
	$(PYTHON) tools/build.py examples
check:
	$(PYTHON) tools/build.py check
check-extended:
	$(PYTHON) tools/run_extended.py
check-asan:
	$(PYTHON) tools/run_extended.py --sanitize address
check-tsan:
	$(PYTHON) tools/run_extended.py --sanitize thread
check-plan:
	$(PYTHON) tools/check_plan.py
	$(PYTHON) tools/check_policy_integrity.py
check-generated:
	$(PYTHON) tools/generate_tuning.py --check
	$(PYTHON) tools/check_native_variants.py
verify-donors:
	$(PYTHON) tools/check_plan.py --verify-donors
verify-evidence:
	$(PYTHON) tools/check_policy_integrity.py --verify-evidence
audit:
	$(PYTHON) tools/audit_build.py
bench:
	$(PYTHON) tools/build.py bench
