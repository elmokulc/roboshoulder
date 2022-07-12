#!/bin/bash
echo "Start Devcontainer post-command process !"

#cd ../workspace/
echo "Installing pre-commit"
echo $CONDA_ENV
${CONDA_ENV}pre-commit install
${CONDA_ENV}pre-commit install-hooks

echo "Installing some additional custom modules"
/opt/conda/envs/roboshoulder/bin/pip install git+https://gitlab.com/symmehub/utils/python_plot_toolbox.git

echo "Devcontainer post-command process done !"
