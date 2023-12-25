# api/userApi.py
import uuid
from flask import Blueprint, request, jsonify
from firebase_admin import firestore
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
import pytesseract
from io import BytesIO

pytesseract.pytesseract.tesseract_cmd = r'c:\Program Files\Tesseract-OCR\tesseract.exe'

db = firestore.client()
user_Ref = db.collection('products')
userAPI = Blueprint('userAPI', __name__)

@userAPI.route('/add', methods=['POST'])
def create():
    try:
        id = str(uuid.uuid4())
        user_Ref.document(id).set(request.json)
        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@userAPI.route('/get', methods=['GET'])
def get_all():
    try:
        users = [doc.to_dict() for doc in user_Ref.stream()]
        return jsonify({"users": users}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@userAPI.route('/get/<string:user_id>', methods=['GET'])
def get_by_field(user_id):
    try:
        query = user_Ref.where("id", '==', user_id)
        user_docs = query.get()

        users_data = [user_doc.to_dict() for user_doc in user_docs]

        if users_data:
            return jsonify({"users": users_data}), 200
        else:
            return jsonify({"error": "Users not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@userAPI.route('/update/<string:user_id>', methods=['PUT'])
def update(user_id):
    try:
        user_Ref.document(user_id).update(request.json)
        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@userAPI.route('/delete/<string:user_id>', methods=['DELETE'])
def delete(user_id):
    try:
        user_Ref.document(user_id).delete()
        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@userAPI.route('/ocr/<string:user_id>', methods=['PUT'])
def ocr(user_id):
    try:
      query = user_Ref.where("id", '==', user_id)
      user_docs = query.get()

      users_data = [user_doc.to_dict() for user_doc in user_docs]
      image = users_data[0].get('imageUrl')
      image +='.png'

      image_pre = preprocess_image(image)
      image_morph = morphology(image_pre)
      text = extract_text(image_morph)
   
      for doc in user_docs:
         user_data = doc.to_dict()
         user_data["ocr"] = True
         user_data["ocr_text"] = text
    
    # Update the document
      doc.reference.update(user_data)
      return jsonify({"users": user_data}), 200
    except Exception as e:
      return jsonify({"error": str(e)}), 500


def preprocess_image(image_path):
    response = requests.get(image_path)
    image = np.array(Image.open(BytesIO(response.content)))
    # img = cv2.imread(image_path)
    
    # Menyamakan ukuran gambar
    img = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

    # Blur gambar menggunakan Gaussian Blur
    blurred_image = cv2.GaussianBlur(img, (1, 1), 0)

    # Konversi gambar ke skala abu-abu
    img_gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)

    # Histogram equalization
    equalized_image = cv2.equalizeHist(img_gray)

    return img_gray

def morphology(image_gray, global_threshold_value=90, morph_kernel_size=(2, 2)):
    # Thresholding
    _, thresholded_image = cv2.threshold(image_gray, global_threshold_value, 255, cv2.THRESH_BINARY)

    # Operasi morfologi
    kernel = np.ones(morph_kernel_size, np.uint8)
    morph_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_ERODE, kernel)

    # Konversi gambar ke format yang dapat ditampilkan dengan Matplotlib
   #  thresholded_image_display = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2RGB)
    opened_image_display = cv2.cvtColor(morph_image, cv2.COLOR_GRAY2RGB)

    return opened_image_display

def extract_text(image, crop_box=(0, 500, 0, 600), config=None):
    cropped_image = image[crop_box[0]:crop_box[1], crop_box[2]:crop_box[3]]

    detected_text = pytesseract.image_to_string(cropped_image, config=config)

    # Split the detected text into lines
    lines = detected_text.splitlines()

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    return lines

