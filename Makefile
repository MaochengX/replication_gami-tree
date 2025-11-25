.PHONY: clear_data
.PHONY: clear_runs
.PHONY: get_data

get_data:
	@uv run src/gami_tree_reproduce/data.py
	@echo "check data folder"

run_simulation:
	@uv run src/gami_tree_reproduce/main.py
	@echo "check assets folder for experiment results"

clear_data:
	@rm data -rf
	@mkdir data
	@echo "🧹 remove data/"


clear_runs:
	@rm assets/simulation_runs -rf
	@echo "🧹 remove asets/simulation_runs/"

# Alias:
clear: clean
