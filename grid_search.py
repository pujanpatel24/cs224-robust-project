"""how to use this file
from home directory, run python scripts/template_run.py 
use flag --print_only to print formatted commands, but not run them
"""

from subprocess import call



# key is the argument name expected by main.py
# value should be a list of options
GRID_SEARCH = {
    'batch-size': [16],
    'lr': [ 1e-6],
    'num-epochs': [7],
    'patience': 4,
    'weight-decay': [0.5],
}


def main():
    for bs in GRID_SEARCH['batch-size']:
        for lr in GRID_SEARCH['lr']:
            for epochs in GRID_SEARCH['num-epochs']:
                for wd in GRID_SEARCH['weight-decay']:
                    for syn in [0, 1]:
                        for bktranslation in [0,1]:
                            model_name = "DATA_grid_search_bs_{}_lr_{}_wd_{}".format(bs, lr, wd)
                            cmd = "python train.py --run-name {} --eval-every 50 --model-path {} ".format(model_name, "save/baseline-early-stop-01/checkpoint")
                            cmd += "--batch-size {} ".format(bs)
                            cmd += "--lr {} ".format(lr)
                            cmd += "--num-epochs {} ".format(epochs)
                            cmd += "--weight-decay {} ".format(wd)
                            cmd += "--do-finetune "
                            if bktranslation:
                                cmd += "--backtranslation "
                            if syn:
                                cmd += "--synonym "
                            cmd += "--visualize-predictions "
                            cmd += "--recompute-features "

                    output = call(cmd, shell=True)
                    f = open("results.txt", "a")
                    f.write("Results for model_name {} is {}".format(model_name, output))
                    f.close()

if __name__ == '__main__':
    main()
