import os
import subprocess
import cv2
import dlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from pyspark.ml.feature import BucketedRandomProjectionLSHModel
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors

# Verify environment variables
hadoop_home = os.environ.get('HADOOP_HOME')
path = os.environ.get('PATH')

print(f"HADOOP_HOME: {hadoop_home}")
print(f"PATH: {path}")

# Verify winutils.exe
try:
    result = subprocess.run(['winutils'], capture_output=True, text=True)
    print("winutils output:")
    print(result.stdout)
    print(result.stderr)
except FileNotFoundError:
    print("winutils.exe not found. Please ensure it is in the PATH.")
except Exception as e:
    print(f"An error occurred: {e}")

# Set environment variables if not already set
if not hadoop_home:
    os.environ['HADOOP_HOME'] = "C:/hadoop"
if 'C:/hadoop/bin' not in path:
    os.environ['PATH'] += os.pathsep + os.path.join(os.environ['HADOOP_HOME'], 'bin')

# Initialize SparkSession with Hadoop configuration
spark = SparkSession.builder \
    .appName("LSH Face Recognition") \
    .config("spark.executor.memory", "6g") \
    .config("spark.driver.memory", "6g") \
    .config("spark.hadoop.home.dir", os.environ['HADOOP_HOME']) \
    .getOrCreate()

def load_lsh_model(model_folder):
    try:
        print(f"Attempting to load LSH model from {model_folder}...")
        loaded_model = BucketedRandomProjectionLSHModel.load(model_folder)
        print("Model loaded successfully.")
        return loaded_model
    except Exception as e:
        print(f"Error loading LSH model: {e}")
        raise

def resize(img, new_size=(224, 224)):
    return cv2.resize(img, new_size)

def detect_largest_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray)
    if not faces:
        print("No faces found in the image.")
        return None
    largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
    x, y, w, h = (largest_face.left(), largest_face.top(), largest_face.width(), largest_face.height())
    face_img = img[y:y+h, x:x+w]
    return face_img

def extract_features(image, model_path):
    model = tf.keras.models.load_model(model_path)
    feature_extraction_model = Model(inputs=model.input, outputs=model.get_layer('global_average_pooling2d').output)
    preprocessed_image = preprocess_input(image)
    processed_image = np.expand_dims(preprocessed_image, axis=0)
    features = feature_extraction_model.predict(processed_image)
    return features

model_path = "C:/Users/phong/Desktop/FinalFacialFunctions/saved_model/my_trained_modelH5.h5"
lsh_model_folder = "C:/Users/phong/Desktop/FinalFacialFunctions/saved_model/saved_lsh_model"
df_path = "C:/Users/phong/Desktop/FinalFacialFunctions/saved_model/df_file.parquet"
input_path = "C:/Users/phong/Desktop/FinalFacialFunctions/Ahmed_Chalabi_0001.jpg"

input_img = cv2.imread(input_path)
resized_img = resize(input_img)
features = extract_features(resized_img, model_path)
key_vector = Vectors.dense(features.flatten())

print("Key Vector here:")
print(key_vector)

lsh_model = load_lsh_model(lsh_model_folder)
print('pass')

df = spark.read.parquet(df_path)
neighbors = lsh_model.approxNearestNeighbors(df, key_vector, numNearestNeighbors=10)

file_name = os.path.basename(input_path)
print("File Name:", file_name)
print("Approximate Nearest Neighbors:")
neighbors.show()

spark.stop()
