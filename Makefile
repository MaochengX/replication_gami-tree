.PHONY: clean-all
.PHONY: data-openml

ID ?= 44048
get_data_openml:
	@DATA_PATH=$$(uv run python -c "from src.gami_tree_reproduce.data import download_data_openml; \
	f=download_data_openml(id=int('$(ID)'));print(f)"); \
	echo "✨ downloaded data into $$DATA_PATH"


clean_data:
	@rm data -rf
	@mkdir data
	@echo "🧹 clean data folder"


# Alias:
clear: clean
