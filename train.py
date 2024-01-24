import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 1: Prepare and Organize Your Dataset
# - Organize your dataset into subfolders (e.g., "dampers", "sparks", "poles") with corresponding images.
# - Use subfolders to separate different classes.

# Step 2: Data Preprocessing
image_size = (224, 224)
batch_size = 32

train_data_dir = "images"
test_data_dir = "images1"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Check if the test directory has data
if len(os.listdir(test_data_dir)) > 0:
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Step 3: Build and Train the Model
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softmax')  # Adjust the number of classes (2 in this case)
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    epochs = 10
    model.fit(train_generator, epochs=epochs)

    # Step 4: Evaluate the Model (Optional)
    if len(test_generator) > 0:
        results = model.evaluate(test_generator)
        print("Test loss, Test accuracy:", results)
    else:
        print("Test data generator has no data.")

    # Step 5: Inference with the Trained Model using Laptop Camera
    camera = cv2.VideoCapture(0)  # Use the default camera (you may need to adjust the camera index)

    while True:
        ret, frame = camera.read()

        if not ret:
            break

        frame = cv2.resize(frame, image_size)
        frame = frame / 255.0  # Normalize pixel values

        # Make predictions using the trained model
        predictions = model.predict(np.expand_dims(frame, axis=0))

        # Get the predicted class label
        predicted_class = np.argmax(predictions)

        # Display the class label on the frame
        label = ['dampers', 'poles'][predicted_class]  # Adjust labels based on your classes
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the frame with the predicted label
        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    camera.release()
    cv2.destroyAllWindows()
else:
    print("Test data directory is empty.")
