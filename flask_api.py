from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Your Hugging Face API token (from environment variable - set this in Render.com)
HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODEL_URL = "https://api-inference.huggingface.co/models/totoro2211/sms-smishing-distilbert"

@app.route('/classify', methods=['POST'])
def classify_sms():
    """
    Endpoint to classify SMS messages
    Expects JSON: {"text": "message to classify"}
    Returns: {"label": "SPAM" or "SAFE", "confidence": 0.95}
    """
    try:
        # Get the text from request
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Call Hugging Face API
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": text,
            "options": {"wait_for_model": True}
        }
        
        response = requests.post(MODEL_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            # Parse the response
            if isinstance(result, list) and len(result) > 0:
                predictions = result[0]
                
                # Find SPAM and SAFE scores
                spam_score = 0.0
                safe_score = 0.0
                
                for pred in predictions:
                    label = pred.get('label', '')
                    score = pred.get('score', 0.0)
                    
                    if label in ['LABEL_1', 'spam', 'SPAM']:
                        spam_score = score
                    elif label in ['LABEL_0', 'safe', 'SAFE']:
                        safe_score = score
                
                # Determine final classification
                if spam_score > safe_score:
                    return jsonify({
                        "label": "SPAM",
                        "confidence": spam_score
                    })
                else:
                    return jsonify({
                        "label": "SAFE",
                        "confidence": safe_score
                    })
            
            return jsonify({"error": "Unexpected response format"}), 500
        
        else:
            return jsonify({
                "error": f"HuggingFace API error: {response.status_code}",
                "details": response.text
            }), 500
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "SMS Classifier API"})

if __name__ == '__main__':
    # Run on all interfaces so it's accessible from your phone
    app.run(host='0.0.0.0', port=5000, debug=True)
