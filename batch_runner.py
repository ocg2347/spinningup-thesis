import os
"""
Sample script to push data to Firebase storage
"""
from firebase_admin import credentials, initialize_app, storage
cred = credentials.Certificate("cnp-actor-f8c04610062d.json")
initialize_app(cred, {"storageBucket": "cnp-actor.appspot.com"})

experiment_cmds = [
    "python spinningup/spinup/algos/pytorch/ppo_1proc/ppo_1proc.py --env Hopper-v3 --exp_name ppo_seed0 --epochs 300 --seed 0",
    "python spinningup/spinup/algos/pytorch/ppo_1proc/ppo_1proc.py --env Hopper-v3 --exp_name ppo_seed11 --epochs 300 --seed 11",
    "python spinningup/spinup/algos/pytorch/ppo_1proc/ppo_1proc.py --env Hopper-v3 --exp_name ppo_seed22 --epochs 300 --seed 22",
    "python spinningup/spinup/algos/pytorch/ppo_cnp_altAct/ppo_cnp.py --env Hopper-v3 --exp_name ppo_cnp_altAct_seed0 --epochs 300 --seed 0",
    "python spinningup/spinup/algos/pytorch/ppo_cnp_altAct/ppo_cnp.py --env Hopper-v3 --exp_name ppo_cnp_altAct_seed11 --epochs 300 --seed 11",
    "python spinningup/spinup/algos/pytorch/ppo_cnp_altAct/ppo_cnp.py --env Hopper-v3 --exp_name ppo_cnp_altAct_seed22 --epochs 300 --seed 22",
]

progress_paths = [
    "spinningup/data/ppo_seed0/ppo_seed0_s0/progress.txt",
    "spinningup/data/ppo_seed11/ppo_seed11_s11/progress.txt",
    "spinningup/data/ppo_seed22/ppo_seed22_s22/progress.txt",
    "spinningup/data/ppo_cnp_altAct_seed0/ppo_cnp_altAct_seed0_s0/progress.txt",
    "spinningup/data/ppo_cnp_altAct_seed11/ppo_cnp_altAct_seed11_s11/progress.txt",
    "spinningup/data/ppo_cnp_altAct_seed22/ppo_cnp_altAct_seed22_s22/progress.txt",
]

for cmd, progress_file in zip(experiment_cmds, progress_paths):
    os.system(cmd)
    # Put your local file path
    try:
        fileName = progress_file
        destBlobName = progress_file.split("/")[2]
        bucket = storage.bucket()
        blob = bucket.blob(destBlobName)
        blob.upload_from_filename(fileName)
    except:
        pass