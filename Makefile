VENV := .venv
PYTHON := $(VENV)/bin/python


.PHONY: sim_data
sim_data:
	$(PYTHON) src/gami_tree_reproduce/data/simulation.py
	@echo "🚀 data simulation completed"

.PHONY: openml
openml:
	$(PYTHON) src/gami_tree_reproduce/data_openml.py
	@echo "🚀 downloaded data from openml"


.PHONY: clear_data
clear_data:
	@rm -rf data
	@rm -rf assets/conf/sim
	@mkdir data
	@echo "🧹 cleared data/ and assets/conf/sim/"



run_simulation:
	$(PYTHON) src/gami_tree_reproduce/main.py
	@echo "🧪 simulation experiment finished"

clear_runs:
	@rm assets/simulation_runs -rf
	@echo "🧹 remove assets/simulation_runs/"

# Alias:
clear: clean
