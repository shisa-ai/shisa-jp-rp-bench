mamba create -n shisa-jp-rp-bench python=3.12 -y
mamba activate shisa-jp-rp-bench
mamba install git-lfs -y

pip install uv
uv pip install -e .
uv pip install choix
