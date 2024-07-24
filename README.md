# skin-Diseases-detection
Skin Disease Detection using Deep Learning and GUI
This repository contains code for predicting skin diseases using deep learning techniques with a graphical user interface (GUI). The model is trained on a dataset of skin disease images and can classify different types of skin diseases with high accuracy.

About
Skin diseases are common and can have a significant impact on an individual's health and quality of life. Early detection and accurate diagnosis of skin conditions are crucial for effective treatment. This project aims to provide a user-friendly tool for automated skin disease prediction using deep learning algorithms.

Code Overview
The main components of the code include:

Loading a pre-trained MobileNet V2 model from TensorFlow Hub
Constructing a deep learning model architecture using Keras
Compiling the model with appropriate optimizer, loss function, and evaluation metrics
Implementing a GUI using tkinter for user interaction
Preprocessing images, predicting diseases, and displaying results in the GUI
Usage

To use this code, follow these steps:
Clone the repository to your local machine:
git clone https://github.com/dusanepooja/skin-disease-detection-gui.git

Install the required dependencies:
pip install -r requirements.txt

Run the main script to launch the GUI application:
python main.py

In the GUI window, click the "Select Image" button to choose an image of a skin lesion.

After selecting an image, click the "Predict" button to predict the disease associated with the image.

The predicted disease label will be displayed below the image.

Contributors
Pooja Dusane

License
This project is licensed under the MIT License - see the LICENSE file for details.
