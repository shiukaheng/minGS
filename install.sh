pip install plyfile lpips pybind11

# Add pybind11 to CPLUS_INCLUDE_PATH
export PYBIND11_DIR=$(python -c "import pybind11; print(pybind11.get_include())")
export CPLUS_INCLUDE_PATH=$PYBIND11_DIR:$CPLUS_INCLUDE_PATH

pip install -e ./submodules/diff-gaussian-rasterization/
pip install -e ./submodules/simple-knn/