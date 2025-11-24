.PHONY: clean-all
.PHONY: data-openml

get_data:
	@python3 src/gami_tree_reproduce/data.py
	@ech "check data folder"


clear_data:
	@rm data -rf
	@mkdir data
	@echo "🧹 remove data/"


clear_runs:
	@rm assets/simulation_runs -rf
	@echo "🧹 remove asets/simulation_runs/"

# Alias:
clear: clean
