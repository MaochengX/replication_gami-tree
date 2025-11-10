.PHONY: clean-all

DATASETS := bike dummy_to_be_added
DEST ?= data/raw

$(DATASETS):
	@echo "📥 Fetching dataset '$@' into $(DEST)..."
	python -c "from pathlib import Path; from src.gami_tree_reproduce.data import get_openml_data; get_openml_data('$@', Path('$(DEST)'))"


clean:
	@echo "🧹 clean data folder (except raw)"
	find data -mindepth 1 -maxdepth 1 ! -name 'raw' -exec rm -rf {} +

clean-all:
	@echo "🧹 clean data folder..."
	rm -rf data
	mkdir data
	mkdir data/raw
	mkdir data/processed


# Alias:
clear: clean
clear-all: clean-all
