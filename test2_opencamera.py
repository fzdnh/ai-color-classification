import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('colormodel.h5')

# Define color labels based on your classes
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

# Open a connection to the camera (0 indicates the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Get the height and width of the frame
    height, width, _ = frame.shape

    # Display a smaller crosshair in the center of the frame
    crosshair_size = 20
    cv2.line(frame, (width // 2 - crosshair_size, height // 2), (width // 2 + crosshair_size, height // 2), (0, 255, 0), 2)
    cv2.line(frame, (width // 2, height // 2 - crosshair_size), (width // 2, height // 2 + crosshair_size), (0, 255, 0), 2)

    # Call the function to predict color on the center of the frame
    center_crop = frame[height // 4:3 * height // 4, width // 4:3 * width // 4]
    predicted_color = predict_color(center_crop)

    # Display the predicted color on the frame
    cv2.putText(frame, f'Predicted Color: {predicted_color}', (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Video Stream with Prediction', frame)

    # Press 'q' to exit the loop and close the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()