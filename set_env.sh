#!/usr/bin/env bash
export HYP_HOME=$(pwd)
export PYTHONPATH="$HYP_HOME:$PYTHONPATH"
python3 -m venv hyp_env
source hyp_env/bin/activate
pip install -r requirements.txt