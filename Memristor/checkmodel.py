# Predicting the inputs
import requests
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

url = 'http://postfiles7.naver.net/20160616_86/ejjun92_1466078439176U1Bsu_PNG/sample_digit.png?type=w580'
response = requests.get(url, stream=True)
img = Image.open(response.raw)
plt.imshow(img, cmap=plt.get_cmap('gray'))


img = np.asarray(img)

print(img.shape)
# img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# img = cv2.resize(img, (28, 28))
# img = cv2.bitwise_not(img)
# plt.imshow(img, cmap=plt.get_cmap('gray'))

# model = lenet5()
# img = img/255
# print(img.shape)
# img = img.reshape(1, 28, 28, 1)
# prediction = model.predict_classes(img)
