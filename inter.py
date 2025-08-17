import onnxruntime as ort
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image 
import torchvision.transforms as transforms

import config

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # convert to grayscale
        transforms.Resize((28, 28)),                # resize to 28x28 pixels
        transforms.ToTensor(),                      # convert to Tensor
        transforms.Normalize((0.5,), (0.5,))        # pixel normalization
    ])

    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print("Image not found")
        return None

    img_tensor = transform(img)

    return img_tensor.unsqueeze(0)


def run_inference_on_local_image():
    IMAGE_PATH = "MNIST - JPG - testing/4/67.jpg"

    # Install ONNX model
    print(f"Đang tải model ONNX từ: {config.ONNX_MODEL_PATH}")
    ort_session = ort.InferenceSession(config.ONNX_MODEL_PATH)
    input_name = ort_session.get_inputs()[0].name

    #(PREPROCESSING)
    input_tensor = preprocess_image(IMAGE_PATH)
    if input_tensor is None:
        return 

    #(EXECUTING)
    ort_inputs = {input_name: input_tensor.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)

    #(POSTPROCESSING)
    logits = ort_outputs[0]
    probabilities = F.softmax(torch.from_numpy(logits), dim=1).numpy()[0]
    predicted_label = np.argmax(probabilities)
    confidence = probabilities[predicted_label]

    print(f"(Predicted)   : {predicted_label}")
    print(f"Accuracy : {confidence:.2%}")

    original_image = plt.imread(IMAGE_PATH)
    plt.imshow(original_image, cmap='gray')
    plt.title(f"Predicted: {predicted_label} | Accuracy: {confidence:.2%}")
    plt.show()

if __name__ == '__main__':
    run_inference_on_local_image()