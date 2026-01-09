VENV := .venv
PYTHON := $(VENV)/bin/python

.PHONY: create_sim_data
create_sim_data:
	$(PYTHON) src/gami_tree_reproduce/data/simulation.py
	@echo "🚀 data simulation completed"

.PHONY: create_plots_sim
create_plots_sim:
	$(PYTHON) src/gami_tree_reproduce/plot/plot_simulation.py
	@echo "📊 simulation data plots created"

.PHONY: show_plots
show_plots:
	@display assets/plots/*.png

.PHONY: create_openml_data
create_openml_data:
	$(PYTHON) src/gami_tree_reproduce/data/data_openml.py
	@echo "🚀 downloaded data from openml"

.PHONY: clear_data
clear_data:
	@rm -rf data
	@rm -rf assets/conf/sim
	@rm -rf assets/plots/data
	@mkdir data
	@mkdir assets/conf/sim
	@mkdir assets/plots/data
	@echo "🧹 cleared data/"
	@echo "🧹 cleared assets/conf/sim"
	@echo "🧹 cleared assets/conf/data"



run_simulation:
	$(PYTHON) src/gami_tree_reproduce/main.py
	@echo "🧪 simulation experiment finished"

clear_runs:
	@rm assets/simulation_runs -rf
	@echo "🧹 remove assets/simulation_runs/"

# Alias:
clear: clean
