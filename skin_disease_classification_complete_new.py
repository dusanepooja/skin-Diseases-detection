# Commented out IPython magic to ensure Python compatibility.
from PIL import Image
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import PIL
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings("ignore")
# %matplotlib inline

'''from google.colab import drive
drive.mount('/content/drive')'''

os.listdir()

dataset_url = r"C:\Users\Hp\Downloads\skin-disease-datasaet-20240512T145037Z-001\skin-disease-datasaet\train_set"

os.listdir(dataset_url)

dataset_url

import pathlib
data_dir=pathlib.Path(dataset_url)
data_dir

cellulitis=list(data_dir.glob('BA- cellulitis/*'))
len(cellulitis)

FU_athlete_foot=list(data_dir.glob('FU-athlete-foot/*'))
len(FU_athlete_foot)

VI_chickenpoxt=list(data_dir.glob('VI-chickenpox/*'))
len(VI_chickenpoxt)

VI_shingles=list(data_dir.glob('VI-shingles/*'))
len(VI_shingles)

FU_nail_fungus=list(data_dir.glob('FU-nail-fungus/*'))
len(FU_nail_fungus)

BA_impetigo=list(data_dir.glob('BA-impetigo/*'))
len(BA_impetigo)

FU_ringworm=list(data_dir.glob('FU-ringworm/*'))
len(FU_ringworm)

PA_cutaneous_larva_migrans=list(data_dir.glob('PA-cutaneous-larva-migrans/*'))
len(PA_cutaneous_larva_migrans)

disease_images_train_dic={
    'cellulitis':list(data_dir.glob('BA- cellulitis/*')),
    'impetigo':list(data_dir.glob('BA-impetigo/*')),
    'athlete-foot':list(data_dir.glob('FU-athlete-foot/*')),
    'nail-fungus':list(data_dir.glob('FU-nail-fungus/*')),
    'ringworm':list(data_dir.glob('FU-ringworm/*')),
    'cutaneous-larva-migrans':list(data_dir.glob('PA-cutaneous-larva-migrans/*')),
    'chickenpox':list(data_dir.glob('VI-chickenpox/*')),
    'shingles':list(data_dir.glob('VI-shingles/*')),
#     'normal':list(data_dir.glob('normal/*')),
}

disease_train_label_dic={
    'cellulitis': 0,
    'impetigo': 1,
    'athlete-foot': 2,
    'nail-fungus': 3,
    'ringworm': 4,
    'cutaneous-larva-migrans':5,
    'chickenpox':6,
    'shingles':7,
#   'normal':8,
}

x_train = []
y_train = []

for image_name, image_paths in disease_images_train_dic.items():
    for image_path in image_paths:
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        image_resize=cv2.resize(img,(224,224))
        x_train.append(image_resize)
        y_train.append(disease_train_label_dic[image_name])

x_train[0].shape

len(x_train)

len(y_train)

y_train=np.array(y_train)
x_train=np.array(x_train)
y_train.shape

dataset_url=r"C:\Users\Hp\Downloads\skin-disease-datasaet-20240512T145037Z-001\skin-disease-datasaet\test_set"

import pathlib
data_dir=pathlib.Path(dataset_url)
data_dir

disease_images_test_dic={
    'cellulitis':list(data_dir.glob('BA- cellulitis/*')),
    'impetigo':list(data_dir.glob('BA-impetigo/*')),
    'athlete-foot':list(data_dir.glob('FU-athlete-foot/*')),
    'nail-fungus':list(data_dir.glob('FU-nail-fungus/*')),
    'ringworm':list(data_dir.glob('FU-ringworm/*')),
    'cutaneous-larva-migrans':list(data_dir.glob('PA-cutaneous-larva-migrans/*')),
    'chickenpox':list(data_dir.glob('VI-chickenpox/*')),
    'shingles':list(data_dir.glob('VI-shingles/*')),
#     'normal':list(data_dir.glob('test_set/normal/*')),
}
disease_test_label_dic={
    'cellulitis': 0,
    'impetigo': 1,
    'athlete-foot': 2,
    'nail-fungus': 3,
    'ringworm': 4,
    'cutaneous-larva-migrans':5,
    'chickenpox':6,
    'shingles':7,
#     'normal':8,
}

x_test = []
y_test = []

for image_name, image_paths in disease_images_test_dic.items():
    for image_path in image_paths:
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        image_resize=cv2.resize(img,(224,224))
        x_test.append(image_resize)
        y_test.append(disease_test_label_dic[image_name])

x_train=np.array(x_train)
y_train=np.array(y_train)
x_test=np.array(x_test)
y_test=np.array(y_test)

x_test.shape

x_train_scaled=x_train/255
x_test_scaled=x_test/255

x_train_scaled[0]

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         input_shape=(224, 224, 3),
                                         trainable=False)

model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(224, 224, 3)),
    keras.layers.Lambda(lambda x: feature_extractor_layer(x)),
    keras.layers.Dense(9, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Assuming x_train_scaled and y_train are your training data
model.fit(x_train_scaled, y_train, epochs=15)



model.save_weights('C:/Users/Hp/Downloads/skin/my_model.weights.h5')

model.summary()

x_test_scaled.shape

model.evaluate(x_test_scaled,y_test)

y_predict=model.predict(x_test_scaled)
# y_predict[0]
y_predicted_labels=[]
for i in y_predict:
    y_predicted_labels.append(np.argmax(i))

y_predicted_labels=np.array(y_predicted_labels)

from sklearn.metrics import confusion_matrix, classification_report
print("Classification Report: \n", classification_report(y_test, y_predicted_labels))

confusion_matrix=tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix, annot=True, fmt='d')

plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

import numpy as np
from keras.models import load_model
from keras.preprocessing import image

model.load_weights('C:/Users/Hp/Downloads/skin/my_model.weights.h5')  # Load your trained model weights

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Assuming your feature extractor expects this size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Function to predict disease given an image
def predict_disease(image_path):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction)
    predicted_disease = next((disease for disease, label in disease_train_label_dic.items() if label == predicted_class), "Unknown")
    return predicted_class, predicted_disease

# Example usage:
image_path = r"C:\Users\Hp\Downloads\skin-disease-datasaet-20240512T145037Z-001\skin-disease-datasaet\test_set\BA-impetigo\10_BA-impetigo (89).jpg"
predicted_class, predicted_disease = predict_disease(image_path)
print("Predicted disease class:", predicted_class)
print("Predicted disease name:", predicted_disease)

