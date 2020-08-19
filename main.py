# imports
from keras.models import load_model
from keras.preprocessing import image
from keras import backend as K
from flask import Flask, jsonify, request
import numpy as np

# Initialize model and flask application
app = Flask(__name__)
model = None


# Define 'prepare image' function to convert image to array
def prep_img(image_path):
    # Initialize size of image for testing
    img_rows, img_cols = 474, 355

    # Change the input shape based on backend
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 3)

    # Get load test image from url
    test_image = image.load_img(image_path, target_size=input_shape)

    # Convert image to array
    image_array = image.img_to_array(test_image)
    image_array = image_array / 255
    test_image = np.array([image_array])

    # Return processed image
    return test_image


# Define 'get prediction' function to return prediction
def get_pred(model, image):
    prediction = model.predict(image)
    water = ['Standing Water', prediction[0][0]]
    false = ['Negative', 1 - prediction[0][0]]
    return [water, false]


# Define predict path
@app.route('/predict', methods=['POST'])
def predict():
    # Initialize return data
    data = {"success": False}

    # Read image path name
    request_data = str(request.data)
    image_path = '/Users/avni/OneDrive - Loudoun County Public Schools/AVNI/Machine Learning/image_data/test_images/' \
                 + request_data[2:-1]

    # Prepare image
    image = prep_img(image_path)

    # Get prediction
    preds = get_pred(model, image)

    # Add prediction to return data
    data["predictions"] = []
    for label, prob in preds:
        response = {"label": label, "probability": float(prob)}
        data["predictions"].append(response)

    # Change success to true
    data["success"] = True

    # Return data
    return jsonify(data)


# If main, load model and start server
if __name__ == "__main__":
    print('Flask Server loading')
    model = load_model('water_cnn_v2')
    print('Model loaded')
    app.run()
    print('Bye Bye')
