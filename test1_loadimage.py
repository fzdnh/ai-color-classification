import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('colormodel.h5')

# Define color labels based on the expected order in your trained model
color_labels = ['Black', 'Blue', 'Brown', 'Green', 'Violet', 'White', 'orange', 'red', 'yellow']

# Function to preprocess input image
def preprocess_image(image):
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize the image to the target size used during training
    image = cv2.resize(image, (150, 150))
    # Normalize pixel values to be between 0 and 1
    image = image / 255.0
    # Expand dimensions to match the input shape expected by the model
    image = np.expand_dims(image, axis=0)
    return image

# Function to predict color using the trained model
def predict_color(image):
    # Preprocess the input image
    processed_image = preprocess_image(image)
    # Make predictions using the loaded model
    predictions = model.predict(processed_image)
    # Get the predicted color label
    predicted_color_label = color_labels[np.argmax(predictions)]
    return predicted_color_label

# Function to open a file dialog and return the selected file path
def open_file_dialog():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path

# Function to update the image and predicted color
def update_image_and_prediction():
    # Get the file path using the file dialog
    file_path = open_file_dialog()
    
    # Read the selected image
    image = cv2.imread(file_path)

    # Call the function to predict color on the loaded image
    predicted_color = predict_color(image)

    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image for display
    image = cv2.resize(image, (300, 300))

    # Convert the image to a PhotoImage object
    img = ImageTk.PhotoImage(Image.fromarray(image))

    # Update the image on the label
    image_label.img = img
    image_label.config(image=img)

    # Update the predicted color text
    predicted_color_label.config(text=f'Predicted Color: {predicted_color}')

# Create the main application window
app = tk.Tk()
app.title("Color Prediction App")

# Create a label for displaying the image
image_label = tk.Label(app)
image_label.pack()

# Create a button to browse and update the image
browse_button = tk.Button(app, text="Browse Image", command=update_image_and_prediction)
browse_button.pack()

# Create a label for displaying the predicted color
predicted_color_label = tk.Label(app, text="Please pick an image")
predicted_color_label.pack()

# Start the application
app.mainloop()
