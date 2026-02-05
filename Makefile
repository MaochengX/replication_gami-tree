VENV := .venv
PYTHON := $(VENV)/bin/python

.PHONY: create_sim_data
create_sim_data:
	@$(PYTHON) src/gami_tree_reproduce/data/simulation.py
	@echo "🚀 data simulation completed and written into data folder"

.PHONY: preprocess_sim_data
preprocess_sim_data:
	@$(PYTHON) src/gami_tree_reproduce/data/preprocess.py
	@echo "🚀 generated preprocessed data in data/preprocesses"


.PHONY: create_plots_sim_data
create_plots_sim_data:
	@$(PYTHON) src/gami_tree_reproduce/plot/plot_simulation.py
	@echo "📊 simulation data plots created and written into assets folder"

.PHONY: show_plots
show_plots:
	@display assets/plots/*.png

.PHONY: create_openml_data
create_openml_data:
	@$(PYTHON) src/gami_tree_reproduce/data/data_openml.py
	@echo "🚀 downloaded data from openml and written into data folder"

.PHONY: clear_data
clear_data:
	@rm -rf data
	@echo "🧹 cleared data/"


.PHONY: run_experiments
run_experiments:
	$(PYTHON) src/gami_tree_reproduce/main.py
	@echo "🧪 experiment simulations finished"
	@echo "	  - configuration settings written into assets folder"

clear_assets:
	@rm assets -rf
	@echo "🧹 remove assets"

# Alias:
clear: clean
