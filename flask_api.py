from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
from huggingface_hub import login
import os

app = Flask(__name__)
CORS(app)

# Login with your Hugging Face token
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    login(token=HF_TOKEN)
    print("Logged in to Hugging Face!")

print("Loading model... this may take a minute...")
try:
    # Load your model using pipeline (exactly like your screenshot!)
    classifier = pipeline("text-classification", model="totoro2211/sms-smishing-distilbert")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    classifier = None

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "SMS Classifier API is running!", 
        "model_loaded": classifier is not None
    })

@app.route('/classify', methods=['POST'])
def classify():
    try:
        if classifier is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Use the classifier (just like your screenshot!)
        result = classifier(text)
        
        # Result format: [{'label': 'LABEL_0', 'score': 0.95}]
        prediction = result[0]
        label_num = prediction['label']
        confidence = prediction['score']
        
        # Map LABEL_0/LABEL_1 to SAFE/SPAM
        label = "SPAM" if label_num == "LABEL_1" else "SAFE"
        
        return jsonify({
            "label": label,
            "confidence": float(confidence)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
