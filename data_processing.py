import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

output_path = os.path.join(os.getcwd(), "processed_data")
os.makedirs(output_path, exist_ok=True)

if os.path.exists(output_path + "\\x_processed.npy") and os.path.exists(output_path + "\\y_processed.npy"):
    x = np.load(os.path.join(output_path, "x_processed.npy"))
    y = np.load(os.path.join(output_path, "y_processed.npy"))
else:
    dataset_path = os.path.join(os.getcwd(), "dataset")
    batch_size = 128

    datagen = ImageDataGenerator(rescale=1./255)

    # Use flow_from_directory to load images in batches
    data_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )

    # Initialize lists to store images and labels
    x = []
    y = []

    # Iterate over the batches and accumulate images and labels
    steps_per_epoch = len(data_generator)
    for i, (images, labels) in enumerate(data_generator):

        # Append the images and labels to the respective lists
        x.append(images)
        y.append(labels)

        # Print the shape of the batch
        print("Batch Shape:", images.shape)

        # Stop the loop after processing all batches
        if i == steps_per_epoch - 1:
            break

    # Concatenate the image and label arrays
    x = np.concatenate(x)
    y = np.concatenate(y)

    np.save(os.path.join(output_path, "x_processed.npy"), x)
    np.save(os.path.join(output_path, "y_processed.npy"), y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Flatten the image arrays if needed
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
