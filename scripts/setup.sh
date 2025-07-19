git submodule update --init --recursive
conda env create -f environment.yml
conda activate thgs
pip install pyg_lib torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
python scripts/setup_dependencies.py build_ext