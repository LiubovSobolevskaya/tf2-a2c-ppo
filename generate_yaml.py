import argparse

import yaml

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
    '--num-seeds',
    type=int,
    default=4,
    help='number of random seeds to generate')
parser.add_argument(
    '--env-name',
    default="PongNoFrameskip-v4",
    help='environment name separated by semicolons')
args = parser.parse_args()



template = "env CUDA_VISIBLE_DEVICES={2} python main.py --env_name {0} --logs_dir logs/{1}/{1}-{2} --seed {2}; "

envs = ["PongNoFrameskip-v4", "BreakoutNoFrameskip-v4", "SeaquestNoFrameskip-v4", "BeamRiderNoFrameskip-v4", "QbertNoFrameskip-v4"]

config = {"session_name": "Atari-games", "windows": []}

for i in range(args.num_seeds):
    panes_list = ""
    for env_name in envs:
        panes_list += template.format(env_name,
                            env_name.split('-')[0].lower(), i)

    config["windows"].append({
        "window_name": "seed-{}".format(i),
        "panes":  [panes_list]
    })

yaml.dump(config, open("run_all_on_T800.yaml", "w"), default_flow_style=False)