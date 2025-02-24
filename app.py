from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)


# Load the TensorFlow SavedModel correctly
MODEL_PATH = "25-Feb021739968558-1000-images-model.h5"  # Ensure correct model path
model = tf.keras.layers.TFSMLayer(MODEL_PATH, call_endpoint="serving_default")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Updated breed list (as provided)
unique_breeds = [
    "affenpinscher", "afghan_hound", "african_hunting_dog", "airedale",
    "american_staffordshire_terrier", "appenzeller", "australian_terrier", 
    "basenji", "basset", "beagle", "bedlington_terrier", "bernese_mountain_dog",
    "black-and-tan_coonhound", "blenheim_spaniel", "bloodhound", "bluetick",
    "border_collie", "border_terrier", "borzoi", "boston_bull", 
    "bouvier_des_flandres", "boxer", "brabancon_griffon", "briard", 
    "brittany_spaniel", "bull_mastiff", "cairn", "cardigan", 
    "chesapeake_bay_retriever", "chihuahua", "chow", "clumber", 
    "cocker_spaniel", "collie", "curly-coated_retriever", "dandie_dinmont", 
    "dhole", "dingo", "doberman", "english_foxhound", "english_setter",
    "english_springer", "entlebucher", "eskimo_dog", "flat-coated_retriever", 
    "french_bulldog", "german_shepherd", "german_short-haired_pointer", 
    "giant_schnauzer", "golden_retriever", "gordon_setter", "great_dane",
    "great_pyrenees", "greater_swiss_mountain_dog", "groenendael", 
    "ibizan_hound", "irish_setter", "irish_terrier", "irish_water_spaniel", 
    "irish_wolfhound", "italian_greyhound", "japanese_spaniel", "keeshond", 
    "kelpie", "kerry_blue_terrier", "komondor", "kuvasz", "labrador_retriever", 
    "lakeland_terrier", "leonberg", "lhasa", "malamute", "malinois", 
    "maltese_dog", "mexican_hairless", "miniature_pinscher", "miniature_poodle",
    "miniature_schnauzer", "newfoundland", "norfolk_terrier", 
    "norwegian_elkhound", "norwich_terrier", "old_english_sheepdog", 
    "otterhound", "papillon", "pekinese", "pembroke", "pomeranian", "pug", 
    "redbone", "rhodesian_ridgeback", "rottweiler", "saint_bernard", "saluki", 
    "samoyed", "schipperke", "scotch_terrier", "scottish_deerhound", 
    "sealyham_terrier", "shetland_sheepdog", "shih-tzu", "siberian_husky", 
    "silky_terrier", "soft-coated_wheaten_terrier", "staffordshire_bullterrier", 
    "standard_poodle", "standard_schnauzer", "sussex_spaniel", "tibetan_mastiff", 
    "tibetan_terrier", "toy_poodle", "toy_terrier", "vizsla", "walker_hound", 
    "weimaraner", "welsh_springer_spaniel", "west_highland_white_terrier", 
    "whippet", "wire-haired_fox_terrier", "yorkshire_terrier"
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded!"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty file received!"}), 400

    # Save file securely
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(file_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Make prediction using TensorFlow SavedModel Layer
    preds_dict = model(img)  # Get dictionary output
    preds = list(preds_dict.values())[0].numpy()[0]  # Extract first tensor
    predicted_class = np.argmax(preds)

    # Ensure index is within range
    if predicted_class < len(unique_breeds):
        predicted_label = unique_breeds[predicted_class].replace("_", " ").title()
    else:
        return jsonify({"error": "Prediction index out of range!"}), 500

    return render_template('index.html', prediction=predicted_label, uploaded_image=filename)

if __name__ == '__main__':
    app.run(debug=True)
