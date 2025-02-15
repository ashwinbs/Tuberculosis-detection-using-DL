import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import getcwd
# import cv2
from PIL import Image



def process():

    model = load_model(getcwd() + '\\model_cnn_new.h5')

    
    image_path=getcwd() + "\\media\\input.png"
    img = Image.open(image_path).convert("L")  # Convert the image to grayscale
    img = img.resize((500, 500))  # Resize the image if necessary
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize the pixel values
    input_data = np.expand_dims(img_array, axis=0)

    # Make the prediction
    predictions = model.predict(input_data)

    # Interpret the prediction
    if predictions >= 0.5:
        result = "Patient has Tuberculosis"
    else:
        result = "Patient Doesn\'t have Tuberculosis"

    # Print the result
    print("Prediction:", result)

    return result
