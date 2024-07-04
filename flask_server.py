import face_recognition
from flask import Flask, request, jsonify
import io
import numpy as np
from PIL import Image

app = Flask(__name__)
@app.route('/update_reference', methods=['POST'])
def update_reference():
    global my_face_encoding
    try:
        file = request.files['image']
        img = Image.open(file.stream)
        img_array = np.array(img)
        face_encodings = face_recognition.face_encodings(img_array)

        if len(face_encodings) > 0:
            my_face_encoding = face_encodings[0]
            return jsonify({"status": True, "message": "Reference image updated"})
        else:
            return jsonify({"status": False, "message": "No face found in reference image"})

    except Exception as e:
        return jsonify({"status": False, "message": str(e)})


# Endpoint to recognize face
@app.route('/recognize', methods=['POST'])
def recognize_face():
    try:
        file = request.files['image']
        img = Image.open(file.stream)
        img_array = np.array(img)
        unknown_face_encodings = face_recognition.face_encodings(img_array)

        if len(unknown_face_encodings) > 0:
            unknown_face_encoding = unknown_face_encodings[0]
        else:
            return jsonify({"status": True, "message": "No Face Detected", "data": 0})

        results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)

        if results[0]:
            return jsonify({"status": True, "message": "Recognition successful", "data": 2})
        else:
            return jsonify({"status": True, "message": "Recognition unsuccessful", "data": 1})

    except Exception as e:
        return jsonify({"status": False, "message": str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
