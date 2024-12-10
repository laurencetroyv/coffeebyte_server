from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
import os
import subprocess
import shutil
import uuid
from datetime import datetime
import socket
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

# Configuration
YOLOV9_PATH = 'yolov9'  # Path to YOLOv9 directory
MODEL_PATH = 'best.pt'  # Path to your model
TEMP_DIR = 'temp'  # Temporary directory for storing images

# Create temp directory if it doesn't exist
os.makedirs(TEMP_DIR, exist_ok=True)


def base64_to_image(base64_string):
    # Remove the base64 prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]

    # Decode base64 string to bytes
    img_bytes = base64.b64decode(base64_string)

    # Convert bytes to numpy array
    nparr = np.frombuffer(img_bytes, np.uint8)

    # Decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


def image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded_string}"


def cleanup_temp_files(temp_dir):
    """Clean up temporary files older than 1 hour"""
    current_time = datetime.now()
    for filename in os.listdir(temp_dir):
        filepath = os.path.join(temp_dir, filename)
        file_time = datetime.fromtimestamp(os.path.getctime(filepath))
        if (current_time - file_time).total_seconds() > 3600:  # 1 hour
            try:
                if os.path.isfile(filepath):
                    os.remove(filepath)
                elif os.path.isdir(filepath):
                    shutil.rmtree(filepath)
            except Exception as e:
                print(f"Error cleaning up {filepath}: {e}")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get base64 image from request
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # Create unique directory for this request
        request_id = str(uuid.uuid4())
        request_dir = os.path.join(TEMP_DIR, request_id)
        os.makedirs(request_dir, exist_ok=True)

        # Save the input image
        image = base64_to_image(data['image'])
        input_path = os.path.join(request_dir, 'input.jpg')
        cv2.imwrite(input_path, image)

        # Run YOLOv9 detect.py
        cmd = [
            'python',
            os.path.join(YOLOV9_PATH, 'detect.py'),
            '--weights', MODEL_PATH,
            '--source', input_path,
            '--project', request_dir,
            '--name', 'exp',
            '--save-txt',  # Save labels
            '--save-conf'  # Save confidences
        ]

        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if process.returncode != 0:
            raise Exception(f"Detection failed: {process.stderr}")

        # Get the results
        output_dir = os.path.join(request_dir, 'exp')
        print(output_dir)
        output_image = os.path.join(output_dir, os.listdir(output_dir)[0])  # Get the first image

        # Convert output image to base64
        annotated_base64 = image_to_base64(output_image)

        # Parse detection results from labels file
        labels_path = os.path.join(output_dir, 'labels\input.txt')
        print("path: ", labels_path)

        detections = []
        if os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                for line in f:
                    values = line.strip().split()
                    detection = {
                        'class': int(values[0]),
                        'confidence': float(values[5]) if len(values) > 5 else 1.0,
                        'bbox': [float(x) for x in values[1:5]]
                    }
                    detections.append(detection)

        # Clean up old temporary files
        cleanup_temp_files(TEMP_DIR)

        return jsonify({
            'results': detections,
            'image': annotated_base64
        })

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up request files
        try:
            shutil.rmtree(request_dir)
        except Exception as e:
            print(f"Error cleaning up request directory: {e}")


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('', port))
            return False
        except socket.error:
            return True


if __name__ == '__main__':
    port = 5001

    # Try different ports if the preferred one is in use
    while is_port_in_use(port) and port < 5010:
        print(f"Port {port} is in use, trying next port...")
        port += 1

    print(f"Starting server on port {port}")

    try:
        # Try starting with gevent
        http_server = WSGIServer(('0.0.0.0', port), app)
        print(f"Server is running at:")
        print(f"Local: http://127.0.0.1:{port}")
        print(f"Network: http://0.0.0.0:{port}")
        print(f"Your IP: http://192.168.1.6:{port}")
        http_server.serve_forever()
    except Exception as e:
        print(f"Failed to start server with gevent: {e}")
        # Fallback to basic Flask server
        print("Falling back to basic Flask server...")
        app.run(host='127.0.0.1', port=port, debug=False)