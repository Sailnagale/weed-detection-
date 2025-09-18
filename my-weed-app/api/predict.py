# /api/predict.py
import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np
import os
import requests
from io import BytesIO
import json
from http.server import BaseHTTPRequestHandler

model_path = os.path.join(os.path.dirname(__file__), 'model', 'weed_binary_classifier_model.tflite')
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, img_height, img_width, _ = input_details[0]['shape']

class_names = ['0.Kena_(Commplina_benghalensio)', '1..Lavhala_(Cyperus_Rotundus)']

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        body = json.loads(post_data)
        image_url = body.get('imageUrl')

        if not image_url:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(json.dumps({"error": "No image URL provided"}).encode())
            return

        try:
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content)).resize((img_width, img_height))
            input_data = np.expand_dims(np.array(img).astype(np.float32), axis=0)
            input_data = input_data / 255.0

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])[0][0]

            if predictions > 0.5:
                predicted_name = class_names[1]
                confidence = predictions * 100
            else:
                predicted_name = class_names[0]
                confidence = (1 - predictions) * 100

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "prediction": predicted_name,
                "confidence": f"{confidence:.2f}%"
            }).encode())

        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())