from PIL import Image

from lit_mlp import LitMLP
from data_module import inference_transforms


class BirdApp:
    def __init__(self):
        self.model = LitMLP.load_from_checkpoint("lit-wandb/1b4xvtnf/checkpoints/epoch=99-step=3199.ckpt")

    def predict(self, x):
        input_tensor = transform_image(x)
        return self.model.predict_app(input_tensor)


def transform_image(infile):
    image = Image.open(infile)
    timg = inference_transforms(image)
    timg.unsqueeze_(0)
    return timg
