import sys

from configs.finetuning_config import finetuning_config
from configs.pretraining_config import pretraining_config
from mura_finetuning.scripts.training_routine import train_model

config = None
for arg in sys.argv:
    if arg == "pretrain":
        config = pretraining_config
        print("Using pretrain config")
        break
    elif arg == "frozen":  # Freeze base and train last layers
        config = finetuning_config
        config["train"]["prefix"] = "frozen"
        config["train"]["train_base"] = False
        config["train"]["learning_rate"]: 0.001
        break
    elif arg == "finetune":  # Finetune whole model with low lr
        config = finetuning_config
        config["train"]["prefix"] = "finetune"
        config["train"]["train_base"] = True
        config["train"]["learning_rate"]: 0.0001
        break

for arg in sys.argv:
    if arg == "--densenet":
        config["model"]["name"] = "densenet"
        break
    elif arg == "--vgg":
        config["model"]["name"] = "vgg"
        break
    elif arg == "--resnet":
        config["model"]["name"] = "resnet"
        break
    elif arg == "--inception":
        config["model"]["name"] = "inception"
        break

print(f'Using {config["train"]["prefix"]} config')
"""config = finetuning_config
config["train"]["prefix"] = "frozen"
config["train"]["train_base"] = False
config["train"]["learning_rate"]: 0.001"""
train_model(config)
