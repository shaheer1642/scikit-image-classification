# built-in libraries
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import joblib

# user-defined libraries
from functions import label_func

model_filename = "rf_model.pkl"
model = joblib.load(model_filename)

def imageDetectionBase64(image):
    image = Image.open(BytesIO(base64.b64decode(image))).convert("RGB")
    image = image.resize((224, 224))  # Adjust the size to match the input size expected by the model
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0) # Reshape the image array to match the input shape expected by the model
    image_array = image_array.reshape(image_array.shape[0], -1) # Flatten the image array

    # Make predictions on the image
    predictions = model.predict(image_array)

    # Print the predicted class
    print("\nPredicted class of single image:", [label_func(el) for el in predictions.tolist()])
    return [label_func(el) for el in predictions.tolist()]
