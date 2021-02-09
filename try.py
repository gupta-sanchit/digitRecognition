from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

im = image.load_img("test/img1.jpg", target_size=(28, 28))
# plt.imshow(im)
# plt.show()

im = image.img_to_array(im)
im = im.astype('float32')

gray = np.dot(im[..., :3], [0.299, 0.587, 0.114])
# gray = im
# plt.imshow(gray, cmap=plt.get_cmap('gray'))
# plt.show()
# reshape the image
img_rows = 28
img_cols = 28
gray = gray.reshape(1, img_rows, img_cols, 1)

# normalize image
gray /= 255

model = load_model("trainedModel/digitRecognizer.h5")
prediction = model.predict(gray)
print(prediction)
print("Number: ", prediction.argmax())
