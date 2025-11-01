import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# ---
# Part 1: Setup
# ---
print("--- Starting Flask Server ---")

# Initialize the Flask application
app = Flask(__name__)

# Define key constants
MODEL_PATH = 'food_freshness_model_v3_finetuned.keras'
IMG_SIZE = (128, 128) # Must be the same size as you used for training
CLASS_NAMES = ['Fresh', 'Rotten'] # From your training output

# ---
# Part 2: Load the Winning Model
# ---
print(f"Loading model from {MODEL_PATH}...")
# We can set compile=False, as the model is already trained
model = load_model(MODEL_PATH, compile=False)
print("Model loaded successfully.")

# ---
# Part 3: Define API Routes
# ---

@app.route('/', methods=['GET'])
def home():
    """Serves the main HTML page (your frontend)."""
    # This tells Flask to find 'index.html' in a folder named 'templates'
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Receives an image and returns a prediction."""
    
    # 1. Check if an image was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400

    try:
        # 2. Pre-process the image
        # Read the file stream into a PIL Image
        img = Image.open(file.stream)
        
        # Convert to RGB (to handle PNGs with transparency)
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        # Resize to the model's expected input size
        img = img.resize(IMG_SIZE)
        
        # Convert the PIL image to a NumPy array
        img_array = image.img_to_array(img)
        
        # Expand dimensions to create a "batch" of 1
        # Shape changes from (128, 128, 3) to (1, 128, 128, 3)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        
        # NOTE: We do NOT need to scale by 1./255 or use preprocess_input.
        # Why? Because those layers are already BUILT-IN to your saved model!
        
        # 3. Make a prediction
        prediction_prob = model.predict(img_array_expanded)[0][0]

        # 4. Format the response
        if prediction_prob > 0.5:
            predicted_class = CLASS_NAMES[1] # 'Rotten'
            confidence = prediction_prob * 100
        else:
            predicted_class = CLASS_NAMES[0] # 'Fresh'
            confidence = (1 - prediction_prob) * 100
        
        # Send the result back as JSON
        return jsonify({
            'prediction': predicted_class,
            'confidence': f"{confidence:.2f}%"
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# ---
# Part 4: Run the Application
# ---
if __name__ == '__main__':
    # Starts the Flask web server
    app.run(debug=True, port=5000)