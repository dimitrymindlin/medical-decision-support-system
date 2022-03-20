from configs.finetuning_config import finetuning_config
from configs.frozen_config import frozen_config
from configs.pretraining_config import pretraining_config
from main.training_routine import train_model
import sys

model_name = "inception"
for arg in sys.argv:
    if arg == "--densenet":
        model_name = "densenet"
        break
    elif arg == "--vgg":
        model_name = "vgg"
        break
    elif arg == "--resnet":
        model_name = "resnet"
        break
    elif arg == "--inception":
        model_name = "inception"
        break

configs = [pretraining_config, frozen_config, finetuning_config]

last_stage = None
last_saved_model_name = None
batch_size = 16

if model_name == "inception":
    configs = configs[1:]
    last_saved_model_timestamp = "2022-03-19--10.48"
    batch_size = 32

for conf in configs:
    conf["model"]["name"] = model_name
    conf["train"]["batch_size"] = batch_size
    conf["test"]["batch_size"] = batch_size
    if conf["train"]["prefix"] != "pretrain":
        conf["train"]["checkpoint_name"] = last_saved_model_timestamp
    last_saved_model_timestamp = train_model(conf, print_console=False)
