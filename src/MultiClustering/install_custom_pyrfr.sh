#!/usr/bin/env bash

git@github.com:sslavian812/random_forest_run.git
cd random_forest_run
mkdir build
cd build
cmake ..
make pyrfr_docstrings
cd python_package
pip3 install . --user
