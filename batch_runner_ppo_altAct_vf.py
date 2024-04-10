import sys
import os
import subprocess
import signal

"""
Sample script to push data to Firebase storage
"""

# from firebase_admin import credentials, initialize_app, storage
# cred = credentials.Certificate("cnp-actor-f8c04610062d.json")
# initialize_app(cred, {"storageBucket": "cnp-actor.appspot.com"})

def signal_handler(sig, frame):
    print("\nCtrl+C pressed. Terminating all subprocesses.")
    for process in subprocesses:
        process.terminate()
    sys.exit(1)

hopper_cmds = [
    # ppo_altAct_awm with r+gammaV alt action adv
    "python spinningup/spinup/algos/pytorch/ppo_altAct_awm/ppo_altAct_awm.py --env Hopper-v3 --exp_name ppo_altAct_awm_vf_hopper --epochs 300 --n_alternative_actions 3 --seed 0  --alt_act_adv val", 
    "python spinningup/spinup/algos/pytorch/ppo_altAct_awm/ppo_altAct_awm.py --env Hopper-v3 --exp_name ppo_altAct_awm_vf_hopper --epochs 300 --n_alternative_actions 3 --seed 11 --alt_act_adv val", 
    "python spinningup/spinup/algos/pytorch/ppo_altAct_awm/ppo_altAct_awm.py --env Hopper-v3 --exp_name ppo_altAct_awm_vf_hopper --epochs 300 --n_alternative_actions 3 --seed 22 --alt_act_adv val", 
    # ppo_altAct_pwm with r+gammaV alt action adv
    "python spinningup/spinup/algos/pytorch/ppo_altAct_pwm/ppo_altAct_pwm.py --env Hopper-v3 --exp_name ppo_altAct_pwm_vf_hopper --epochs 300 --n_alternative_actions 3 --seed 0  --alt_act_adv val",
    "python spinningup/spinup/algos/pytorch/ppo_altAct_pwm/ppo_altAct_pwm.py --env Hopper-v3 --exp_name ppo_altAct_pwm_vf_hopper --epochs 300 --n_alternative_actions 3 --seed 11 --alt_act_adv val",
    "python spinningup/spinup/algos/pytorch/ppo_altAct_pwm/ppo_altAct_pwm.py --env Hopper-v3 --exp_name ppo_altAct_pwm_vf_hopper --epochs 300 --n_alternative_actions 3 --seed 22 --alt_act_adv val",
]

walker_cmds = [
   # ppo_altAct_awm with r+gammaV alt action adv
    "python spinningup/spinup/algos/pytorch/ppo_altAct_awm/ppo_altAct_awm.py --env Walker2d-v3 --exp_name ppo_altAct_awm_vf_walker --epochs 750 --n_alternative_actions 3 --seed 0  --alt_act_adv val", 
    "python spinningup/spinup/algos/pytorch/ppo_altAct_awm/ppo_altAct_awm.py --env Walker2d-v3 --exp_name ppo_altAct_awm_vf_walker --epochs 750 --n_alternative_actions 3 --seed 11 --alt_act_adv val", 
    "python spinningup/spinup/algos/pytorch/ppo_altAct_awm/ppo_altAct_awm.py --env Walker2d-v3 --exp_name ppo_altAct_awm_vf_walker --epochs 750 --n_alternative_actions 3 --seed 22 --alt_act_adv val", 
    # ppo_altAct_pwm with r+gammaV alt action adv
    "python spinningup/spinup/algos/pytorch/ppo_altAct_pwm/ppo_altAct_pwm.py --env Walker2d-v3 --exp_name ppo_altAct_pwm_vf_walker --epochs 750 --n_alternative_actions 3 --seed 0  --alt_act_adv val",
    "python spinningup/spinup/algos/pytorch/ppo_altAct_pwm/ppo_altAct_pwm.py --env Walker2d-v3 --exp_name ppo_altAct_pwm_vf_walker --epochs 750 --n_alternative_actions 3 --seed 11 --alt_act_adv val",
    "python spinningup/spinup/algos/pytorch/ppo_altAct_pwm/ppo_altAct_pwm.py --env Walker2d-v3 --exp_name ppo_altAct_pwm_vf_walker --epochs 750 --n_alternative_actions 3 --seed 22 --alt_act_adv val",
]

experiment_cmds = hopper_cmds + walker_cmds

n_processes = 3

cmd_chunks = [experiment_cmds[i:i + n_processes] for i in range(0, len(experiment_cmds), n_processes)]

import time

for i, chunk in enumerate(cmd_chunks):
    subprocesses = []
    for cmd in chunk:
        print(cmd)
        # spawn a new subprocess with null stdout but stderr to the main process
        process=subprocess.Popen(
            cmd.split(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )
        time.sleep(3.5)
        subprocesses.append(process)
    # Set up the signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    try:
        # Wait for all subprocesses to finish
        for process in subprocesses:
            return_code = process.wait()
            print("Return code:", return_code)
        print(f"All subprocesses finished for the chunk{i}.")
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)  # Call the signal handler manually

# os.system("sudo shutdown now")  # Shutdown the machine after all experiments are done