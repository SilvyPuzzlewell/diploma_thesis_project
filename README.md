Code and data we run the experiments on. The environment is based on the code provided with Gri and on the experiment setup for GTS
used in the shallow heuristics paper. We expanded their work for our needs.

Run "source setup.sh" to prepare venv, install requirements and additional dependencies.
run create_data.py for the data preprocessing procedure. The WP500CZ data aren't public, we don't have permission to share them.

For running Gri, prepare config file in stuff/griffiths_solver_vunedited/; examples for running mawps folds are in mawps_configs.
Gri can then be run by "python run_gri.py {relative path from root to the config}", for example "python run_gri.py stuff/griffiths_solver_vunedited/mawps_configs/config0.yaml"

For GTS, the configs are prepared in stuff/gts/configs
Explanation is in stuff/gts/args.py
Run example then is "python run_gts.py -config=debug2.yaml"


 

