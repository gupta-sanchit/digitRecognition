import cv2
import numpy as np
from tensorflow.python.keras.models import load_model

img = cv2.imread("test/img2.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
gray = cv2.resize(gray, (28, 28))
gray = gray.reshape(1, 28, 28, 1)


model = load_model("trainedModel/digitRecognizer.h5")
result = np.argmax(model.predict(gray))
print(result)