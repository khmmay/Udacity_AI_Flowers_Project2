import argparse
import numpy as np
import json
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

from utilities import process_image,predict

image_size = 224

parser = argparse.ArgumentParser(description='Predict flower names')
parser.add_argument('image_path',action="store")
parser.add_argument('model_path',action="store")
parser.add_argument('--top_K',action="store",dest='top_K',type=int,default=1)
parser.add_argument('--category_names',action="store",dest='category_names',default='./label_map.json')

results=parser.parse_args()

reloaded_keras_model_from_SavedModel = tf.keras.models.load_model(results.model_path,custom_objects={'KerasLayer': hub.KerasLayer})
model=reloaded_keras_model_from_SavedModel


with open(results.category_names, 'r') as f:
    class_names = json.load(f)


probs, classes=predict(results.image_path,image_size,model,results.top_K)

print('\n')
for i in range(0,results.top_K):
    print('{}: {}'.format(class_names[str(classes[i]+1)],probs[i]))