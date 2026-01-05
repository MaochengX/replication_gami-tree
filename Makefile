VENV := .venv
PYTHON := $(VENV)/bin/python


.PHONY: sim_data
sim_data:
	$(PYTHON) src/gami_tree_reproduce/data/simulation.py

.PHONY: openml
openml:
	$(PYTHON) src/gami_tree_reproduce/data_openml.py


.PHONY: clear_data
clear_data:
	@rm -rf data
	@mkdir data
	@echo "🧹 cleared /data"



run_simulation:
	$(PYTHON) src/gami_tree_reproduce/main.py

clear_runs:
	@rm assets/simulation_runs -rf
	@echo "🧹 remove assets/simulation_runs/"

# Alias:
clear: clean
