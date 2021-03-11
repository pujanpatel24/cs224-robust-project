"""how to use this file
from home directory, run python scripts/template_run.py 
use flag --print_only to print formatted commands, but not run them
"""
import os, argparse
from subprocess import call
from sklearn.model_selection import ParameterGrid
from datetime import date


# key is the argument name expected by main.py
# value should be a list of options
GRID_SEARCH = {
    'batch-size': [2,4,8,16],
    'lr': [5e-5,1e-4,1e-3, 1e-2],
    'num-epochs': [40],
    'num-layers': [0,1,2]
}


def main():
    for bs in [2,4,8,16]:
        for lr in [5e-5,1e-4,1e-3, 1e-2]:
            for epochs in [40]:
                for layers in [-1,0,1,2]:
                    for backtranslation in [False, True]:
                        for synonym in [False, True]:
                            model_name = "grid_search_bs_{}_lr_{}_num_layers_{}_bt_{}_syn_{}".format(bs, lr, layers, backtranslation, synonym)
                            cmd = "python train.py --run-name {} --eval-every 150 --model-path {} ".format(model_name, "save/baseline-early-stop-01")
                            cmd += "--batch-size {} ".format(bs)
                            cmd += "--lr {} ".format(lr)
                            cmd += "--num-epochs {} ".format(epochs)
                            if layers == -1:
                                cmd += "--do-train "
                            else:
                                cmd += "--do-finetune "
                                cmd += "--num-layers {} ".format(layers)
                            if backtranslation:
                                cmd += "--backtranslation "
                            if synonym:
                                cmd += "--synonym "

                            output = call(cmd, shell=True)
                            f = open("results.txt", "a")
                            f.write("Results for model_name {} is {}".format(model_name, output))
                            f.close()

if __name__ == '__main__':
    main()
