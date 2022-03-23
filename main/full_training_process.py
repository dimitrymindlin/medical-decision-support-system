from configs.finetuning_config import finetuning_config
from configs.frozen_config import frozen_config
from configs.pretraining_config import pretraining_config
from main.training_routine import train_model
import sys

model_name = "inception"
batch_size = None
for arg in sys.argv:
    if arg == "--densenet":
        model_name = "densenet"
        batch_size = 8
        break
    elif arg == "--inception":
        model_name = "inception"
        break

configs = [pretraining_config, frozen_config, finetuning_config]

last_stage = None
last_saved_model_name = None

if model_name == "inception":
    configs = configs[2:]
    last_saved_model_timestamp = "2022-03-21--10.17"


for conf in configs:
    conf["model"]["name"] = model_name
    if batch_size:
        conf["train"]["batch_size"] = batch_size
        conf["test"]["batch_size"] = batch_size
    if conf["train"]["prefix"] != "pretrain":
        conf["train"]["checkpoint_name"] = last_saved_model_timestamp
    last_saved_model_timestamp = train_model(conf, print_console=False)
