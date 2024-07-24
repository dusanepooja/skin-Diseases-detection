import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
from keras.models import load_model

# Load the trained model
model = load_model('my_model.weights.h5')

# Dictionary mapping disease labels to their names
disease_names = {
    0: 'cellulitis',
    1: 'impetigo',
    2: 'athlete-foot',
    3: 'nail-fungus',
    4: 'ringworm',
    5: 'cutaneous-larva-migrans',
    6: 'chickenpox',
    7: 'shingles'
}

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def predict_disease(image_path):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction)
    predicted_disease = disease_names.get(predicted_class, "Unknown")
    return predicted_disease

def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        predicted_disease = predict_disease(file_path)
        messagebox.showinfo("Prediction Result", f"The predicted disease is: {predicted_disease}")

# Create the main application window
root = tk.Tk()
root.title("Skin Disease Predictor")

# Create a button to select an image
button = tk.Button(root, text="Select Image", command=select_image)
button.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
