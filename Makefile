
.PHONY: sim_data
sim_data:
	@uv run python3 src/gami_tree_reproduce/data_simulation.py size=50000 cor=0 --filenameprefix=sim1
	@uv run python3 src/gami_tree_reproduce/data_simulation.py size=50000 cor=1 --filenameprefix=sim2
	@uv run python3 src/gami_tree_reproduce/data_simulation.py size=5000 cor=0 --filenameprefix=sim3
	@uv run python3 src/gami_tree_reproduce/data_simulation.py size=5000 cor=1 --filenameprefix=sim4

.PHONY: clear_sim_data
clear_sim_data:
	@find data -maxdepth 1 -type f -name "sim*" -delete
	@rm assets/conf/data -rf



.PHONY: openml
openml:
	@uv run src/gami_tree_reproduce/data_openml.py

.PHONY: clear_openml_data
clear_openml:
	@find data -maxdepth 1 -type f ! -name "sim*" -delete


.PHONY: clear_data
clear_data: clear_sim_data clear_openml_data


run_simulation:
	@uv run src/gami_tree_reproduce/main.py

clear_runs:
	@rm assets/simulation_runs -rf
	@echo "🧹 remove asets/simulation_runs/"

# Alias:
clear: clean
