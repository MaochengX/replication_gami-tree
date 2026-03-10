.PHONY: sim_data clear_sim_data clear_data training_r training_c training clear_training figure_r figure_c figures clear_figures clear	

sim_data:
	@uv run python3 src/gami_tree_reproduce/data/simulation.py size=500000 cor=0 --filenameprefix=sim1
	@uv run python3 src/gami_tree_reproduce/data/simulation.py size=500000 cor=0.5 --filenameprefix=sim2
	@uv run python3 src/gami_tree_reproduce/data/simulation.py size=50000 cor=0 --filenameprefix=sim3
	@uv run python3 src/gami_tree_reproduce/data/simulation.py size=50000 cor=0.5 --filenameprefix=sim4

clear_sim_data:
	@find data -maxdepth 1 -type f -name "sim*" -delete
	@rm -rf data
	@echo "removed simulation data"

clear_data: clear_sim_data 

training_r:
	@uv run python -m src.gami_tree_reproduce.training_r
	@echo "regression training completed"

training_c:
	@uv run python -m src.gami_tree_reproduce.training_c
	@echo "classification training completed"

training: training_r training_c

clear_training:
	@rm -rf src/gami_tree_reproduce/cache
	@echo "removed all cache files"

figure_r:
	@uv run python -m src.gami_tree_reproduce.figure_r
	@echo "regression figures generated"

figure_c:
	@uv run python -m src.gami_tree_reproduce.figure_c
	@echo "classification figures generated"

figures: figure_r figure_c

clear_figures:
	@rm -rf figures_regression
	@rm -rf figures_classification
	@echo "removed all figure folders"

clear: clear_data clear_figures