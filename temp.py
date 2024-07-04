from flask import Flask, request, jsonify
import face_recognition
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/recognize', methods=['POST'])
def recognize_face():
    if 'reference_image' not in request.files or 'image' not in request.files:
        return jsonify({"error": "Please provide both reference_image and image files"}), 400
    reference_image = request.files['reference_image']
    image = request.files['image']
    reference_image_path = secure_filename(reference_image.filename)
    image_path = secure_filename(image.filename)
    reference_image.save(reference_image_path)
    image.save(image_path)
    try:
        reference_picture = face_recognition.load_image_file(reference_image_path)
        reference_face_encoding = face_recognition.face_encodings(reference_picture)[0]

        unknown_picture = face_recognition.load_image_file(image_path)
        unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]

        results = face_recognition.compare_faces([reference_face_encoding], unknown_face_encoding)

        os.remove(reference_image_path)
        os.remove(image_path)

        if results[0] == True:
            return jsonify({"data": 2})  # Face recognized
        else:
            return jsonify({"data": 1})  # Unrecognized face
    except IndexError:
        return jsonify({"data": 0})  # No face detected
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
