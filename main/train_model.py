import sys

from configs.direct_training_config import direct_training_config
from configs.finetuning_config import finetuning_config
from configs.frozen_config import frozen_config
from configs.pretraining_config import pretraining_config
from main.training_routine import train_model

config = finetuning_config
for arg in sys.argv:
    if arg == "--pretrain":
        config = pretraining_config
        break
    elif arg == "--frozen":  # Freeze base and train last layers
        config = frozen_config
        break
    elif arg == "--finetuning":  # Finetune whole model with low lr
        config = finetuning_config
        break
    elif arg == "--direct":
        config = direct_training_config
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


"""config = finetuning_config
config["train"]["prefix"] = "frozen"
config["train"]["train_base"] = False
config["train"]["learning_rate"]: 0.001"""
"""config = direct_training_config
config["train"]["epochs"] = 1

print(f'Using {config["train"]["prefix"]} config')"""
train_model(config)
