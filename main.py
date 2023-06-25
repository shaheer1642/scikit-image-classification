# built-in libraries
import os
import numpy as np
from sklearn.metrics import accuracy_score
from PIL import Image

# user-defined libraries
from model import model
from data_processing import X_test, y_test
from functions import label_func

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the accuracy of the classifier on test set
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)
print("\ntest-set:", [label_func(el) for el in y_test.tolist()])
print("\npredictions:", [label_func(el) for el in predictions.tolist()])

# Load and preprocess a single image
image_path = os.path.join(os.getcwd(), "image.jpg")  # Get the image.jpg from root dir
image = Image.open(image_path).convert("RGB")
image = image.resize((224, 224))  # Adjust the size to match the input size expected by the model
image_array = np.array(image)
image_array = image_array / 255.0  # Normalize the image
image_array = np.expand_dims(image_array, axis=0) # Reshape the image array to match the input shape expected by the model
image_array = image_array.reshape(image_array.shape[0], -1) # Flatten the image array

# Make predictions on the image
predictions = model.predict(image_array)

# Print the predicted class
print("\nPredicted class of single image:", [label_func(el) for el in predictions.tolist()])