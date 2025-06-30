import os
import sys
import cv2
import numpy as np
import dlib
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Set SPARK and HADOOP paths early
SPARK_HOME = os.path.normpath(os.path.join(settings.BASE_DIR, 'spark-3.2.1-bin-hadoop3.2'))
HADOOP_HOME = os.path.normpath(os.path.join(settings.BASE_DIR, 'hadoop-3.2.1'))
os.environ['SPARK_HOME'] = SPARK_HOME
os.environ['HADOOP_HOME'] = HADOOP_HOME
os.environ['PATH'] += os.pathsep + os.path.join(HADOOP_HOME, 'bin')

# Append PySpark paths to sys.path
sys.path.append(os.path.join(SPARK_HOME, 'python'))
# NOTE: Check the actual name of your Py4J zip file
py4j_zip = [f for f in os.listdir(os.path.join(SPARK_HOME, 'python', 'lib')) if f.startswith('py4j') and f.endswith('.zip')]
if py4j_zip:
    sys.path.append(os.path.join(SPARK_HOME, 'python', 'lib', py4j_zip[0]))

from pyspark.sql import SparkSession
from pyspark.ml.feature import BucketedRandomProjectionLSHModel
from pyspark.ml.linalg import Vectors

# Model paths
model_path = os.path.join(settings.BASE_DIR, 'saved_model', 'my_trained_modelH5.h5')
lsh_model_folder = os.path.join(settings.BASE_DIR, 'saved_model', 'saved_lsh_modelv2')
df_path = os.path.join(settings.BASE_DIR, 'saved_model', 'df_facesv2.parquet')

# Globals
spark = None
lsh_model = None
df = None
model = None

def load_lsh_model(model_folder):
    try:
        return BucketedRandomProjectionLSHModel.load(model_folder)
    except Exception as e:
        print(f"Error loading LSH model: {e}")

def resize(img, new_size=(224, 224)):
    return cv2.resize(img, new_size)

def extract_features(image, model_path):
    global model
    if model is None:
        model = tf.keras.models.load_model(model_path)
    feature_model = Model(inputs=model.input, outputs=model.get_layer('global_average_pooling2d').output)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return feature_model.predict(image)

def detect_largest_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray)
    if not faces:
        print("No faces detected.")
        return None
    largest_face = max(faces, key=lambda r: r.width() * r.height())
    x, y, w, h = largest_face.left(), largest_face.top(), largest_face.width(), largest_face.height()
    return img[y:y+h, x:x+w]

def upload_image(request):
    return render(request, 'upload_image.html')

@csrf_exempt
def process_image_and_get_neighbors(request):
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_image = request.FILES['image']
        image_path = handle_uploaded_image(uploaded_image)
        try:
            neighbors_paths = get_neighbors(image_path)
            image_url = os.path.join(settings.MEDIA_URL, 'uploads', uploaded_image.name)
            return render(request, 'show_results.html', {
                'file_name': uploaded_image.name,
                'neighbors': neighbors_paths,
                'image_url': image_url
            })
        except Exception as e:
            print(f"Error processing image: {e}")
            return render(request, 'upload_image.html', {'error': 'An error occurred while processing the image.'})
    else:
        return render(request, 'upload_image.html')

def handle_uploaded_image(uploaded_image):
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    path = os.path.join(upload_dir, uploaded_image.name)
    with open(path, 'wb+') as dest:
        for chunk in uploaded_image.chunks():
            dest.write(chunk)
    print(f"Image written to: {path}")
    return path

def get_neighbors(image_path):
    global spark, lsh_model, df

    if spark is None:
        spark = SparkSession.builder \
            .appName("LSH Face Recognition") \
            .config("spark.driver.memory", "6g") \
            .getOrCreate()

    input_img = cv2.imread(image_path)
    if input_img is None:
        raise ValueError("Could not read image")
    input_img = detect_largest_face(input_img)
    if input_img is None:
        raise ValueError("No face detected")

    resized_img = resize(input_img)
    features = extract_features(resized_img, model_path)
    vector = Vectors.dense(features.flatten())

    if lsh_model is None:
        lsh_model = load_lsh_model(lsh_model_folder)
    if df is None:
        df = spark.read.parquet(df_path)

    neighbors = lsh_model.approxNearestNeighbors(df, vector, 10)
    return [row['img_path'] for row in neighbors.collect()]
