#from tf_explain.core.smoothgrad import SmoothGrad
#from tf_explain.callbacks.vanilla_gradients import VanillaGradientsCallback
#from tf_explain.core.grad_cam import GradCAM
from matplotlib import pyplot as plt
import numpy as np

from keras_explain.lrp import LRP
import tensorflow as tf
from configs.wrist_xray_config import wrist_xray_config
from wrist_xray_finetuning.dataloader import WristXrayDataset
from wrist_xray_finetuning.model.wrist_xray_model import WristXrayDenseNet

config = wrist_xray_config
dataset = WristXrayDataset(config)
data = dataset.ds_test.take(10)

train_base = config['train']['train_base']
model = WristXrayDenseNet(config, train_base=train_base).model()
model.load_weights("../../checkpoints/mura/best/cp.ckpt")

for index, example in enumerate(data):
    print(index)
    image, label = example[0].numpy(), example[1].numpy()
    image = tf.keras.preprocessing.image.img_to_array(image)

    """# Start explainer
    print(image.shape)
    explainer = GradCAM()
    grid = explainer.explain(([image], None), model, class_index=0)
    print("Label: %d" % label)
    explainer.save(grid, "./explainable_ai", output_name=f"grad_cam_{index}.png")"""

    explainer = LRP(model)
    exp = explainer.explain(image, label)