from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ---------------------------
# BASE PATH (IMPORTANT for Render)
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------
# MODEL LOAD (FIXED FOR RENDER)
# ---------------------------
MODEL_PATH = os.path.join(BASE_DIR, "models", "my-model")

model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures["serving_default"]

# ---------------------------
# UPLOAD FOLDER (SAFE PATH)
# ---------------------------
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------
# CLASS LABELS
# ---------------------------
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

# ---------------------------
# ROUTES
# ---------------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check file
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "Empty file"}), 400

        # Save file safely
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Preprocess image
        img = tf.keras.preprocessing.image.load_img(file_path, target_size=(224, 224))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # TensorFlow inference (FIXED)
        preds_dict = infer(tf.constant(img, dtype=tf.float32))
        preds = list(preds_dict.values())[0].numpy()[0]

        predicted_class = int(np.argmax(preds))

        if predicted_class >= len(unique_breeds):
            return jsonify({"error": "Prediction index out of range"}), 500

        predicted_label = unique_breeds[predicted_class].replace("_", " ").title()

        return render_template(
            'index.html',
            prediction=predicted_label,
            uploaded_image=filename
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------
# MAIN (IMPORTANT FOR LOCAL ONLY)
# ---------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
