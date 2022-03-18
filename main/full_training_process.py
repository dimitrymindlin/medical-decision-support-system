from configs.finetuning_config import finetuning_config
from configs.frozen_config import frozen_config
from configs.pretraining_config import pretraining_config
from main.training_routine import train_model

configs = [pretraining_config, frozen_config, finetuning_config]

last_stage = None
last_saved_model_name = None

for conf in configs:
    if conf["train"]["prefix"] != "pretrain":
        conf["train"]["checkpoint_name"] = last_saved_model_timestamp
    last_saved_model_timestamp = train_model(conf, print_console=False)
