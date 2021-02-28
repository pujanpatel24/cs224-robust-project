## Starter code for robustqa track
Team: Andrew Hwang, Pujan Patel, Max Pike
- Download datasets from [here](https://drive.google.com/file/d/1Fv2d30hY-2niU7t61ktnMsi_HUXS6-Qx/view?usp=sharing)
- Setup environment with `conda env create -f environment.yml`
- Train a baseline MTL system with `python train.py --do-train --eval-every 2000 --run-name baseline`
- Evaluate the system on test set with `python train.py --do-eval --sub-file mtl_submission.csv --save-dir save/baseline-01`
- Upload the csv file in `save/baseline-01` to the test leaderboard. For the validation leaderboard, run `python train.py --do-eval --sub-file mtl_submission_val.csv --save-dir save/baseline-01 --eval-dir datasets/oodomain_val`

Table of our own experiments overtime with hyperparameters chosen: 
https://docs.google.com/spreadsheets/d/1ZqgFEHeP8G1bj22o0WHS_g3ieMre28cU5rsFM9O8m34/edit#gid=0
