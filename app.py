from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Create the 'temp' directory if it doesn't exist
if not os.path.exists('temp'):
    os.makedirs('temp')


# Load your deep learning model
model = load_model('lung_cancer_prediction_model.h5')

# Define route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Define route for handling image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file
    img = request.files['file']

    # Save the uploaded image to a temporary file
    img_path = 'temp/temp_img.jpg'
    img.save(img_path)

    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(299, 299))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.inception_v3.preprocess_input(img)  # Normalize pixel values

    # Perform prediction using your deep learning model
    prediction = model.predict(img)

    # Get the predicted class label and probability
    if prediction[0][0] > 0.5:
        prediction_label = 'Cancerous'
    else:
        prediction_label = 'Non-cancerous'
    
    prediction_probability = prediction[0][0]
    

    # Delete the temporary image file
    os.remove(img_path)

    # Render the prediction result template and pass the prediction result as context
    return render_template('result.html', prediction=prediction_label, probability=prediction_probability)

if __name__ == '__main__':
    app.run(debug=True)
