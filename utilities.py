import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import numpy as np
from PIL import Image

def process_image(image,image_size):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image

def predict(image_path, image_size,model, top_k):
    probs=[]
    classes=[]
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image,image_size)
    processed_test_image = np.expand_dims(processed_test_image,axis=0)
    print(processed_test_image.shape)
    ps=np.squeeze(model.predict(processed_test_image))
    classes = (-ps).argsort()[:top_k]
    probs=ps[classes]
    return probs, classes