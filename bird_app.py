import json
import numpy as np
import onnxruntime
from PIL import Image


class BirdApp:
    def __init__(self):
        self.onnx_session = onnxruntime.InferenceSession("model4app.onnx")
        self.img_class_map = get_img_class_map()

    def predict(self, x):
        input_tensor = transform_image(x)
        onnx_inputs = {self.onnx_session.get_inputs()[0].name: input_tensor}
        img_label = self.onnx_session.run(None, onnx_inputs)[0].argmax()
        return {'class_id': int(img_label), 'class_name': self.img_class_map[str(img_label)]}


def transform_image(infile) -> np.array:
    image = (Image
             .open(infile)
             .resize((224, 224))
             )
    return np.expand_dims(np.array(image, dtype=np.float32), 0).transpose([0, 3, 1, 2])


def get_img_class_map():
    with open('index_to_name.json') as f:
        img_class_map = json.load(f)
    return img_class_map
