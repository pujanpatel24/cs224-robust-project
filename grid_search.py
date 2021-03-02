"""how to use this file
from home directory, run python scripts/template_run.py 
use flag --print_only to print formatted commands, but not run them
"""
import os, argparse
from subprocess import call
from sklearn.model_selection import ParameterGrid
from datetime import date

# any special experiment name to name output folders
EXP_NAME = None

# key is the argument name expected by main.py
# value should be a list of options
GRID_SEARCH = {
    'lr': [1e-3, 1e-2],
    'pretrained': [True, False]
}

# key is the argument name expected by main.py
# value should be a single value
OTHER_PARAMETERS = {
    'dataset_name': "test",
    'gpus': 1 
}

def main(args):
    grid = ParameterGrid(GRID_SEARCH)
    input(f"This will result in {len(list(grid))} exps. OK?")
    
    name = f"_{EXP_NAME}" if EXP_NAME else ""
    for grid_params in grid:
        exp = f"{date.today()}{name}"
        cmd = "python main.py train"
        
        # add grid search params
        for key, val in grid_params.items():
            exp += f"_{key}={val}"
            cmd += f" --{key} {val}"
        
        # add other params
        for key, val in OTHER_PARAMETERS.items():
            cmd += f" --{key} {val}"
        cmd += f" --exp_name {exp}"
        
        if args.print_only:
            print(cmd)
        else:
            print(f"Running command: {cmd}")
            call(cmd, shell=True)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--print_only', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
