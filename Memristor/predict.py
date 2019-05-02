from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read Image


# def prediction(pred_image):
#     img = Image.open(pred_image)

#     # Convert to a nump array
#     img = np.asarray(img)
#     # Convert the image to grey scale
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#     # Resize the image
#     img = cv2.resize(img, (28, 28))

# def visualize_layer_output():
#     # Visualizing the learned features
#     layer1 = Model(inputs = model.layers[0].input, outputs=model.layers[0].output)
#     layer2 = Model(inputs = model.layers[0].input, outputs=model.layers[2].output)

#     visual_layer1 = layer1.predict(img)
#     visual_layer2 = layer2.predict(img)

#     print(visual_layer1.shape)
#     print(visual_layer2.shape)
