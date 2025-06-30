FinalFacialFunctions – Face Recognition System
A deep learning-powered face recognition system built with TensorFlow, Django, Spark MLlib, and EfficientNetB0, developed for the TDTU Student Scientific Research Competition 2023–2024.

Built mostly solo in Year 2 of my studies at Ton Duc Thang University, handling training, model tuning, Django backend, Spark-based indexing, and image processing end-to-end.

Features

✅ Face detection using dlib (frontal face detector).
✅ Feature extraction via EfficientNetB0 pretrained model.
✅ Fast similarity search using Spark MLlib LSH (Locality-Sensitive Hashing).
✅ Web UI (Django) for uploading images and visualizing similar faces.
✅ Trained model and face embeddings pre-saved in Parquet format.
✅ Supports image uploads, processing, and matching against a known dataset.

Technologies Used


Component
Tech Stack



Backend Server
Django 3.2.x


Deep Learning
TensorFlow + Keras (EfficientNetB0) trained from the dataset Labeled Faces in the Wild (LFW


Face Detection
dlib


Vector Indexing
Spark 3.2.1 + MLlib + LSH


Storage Format
Parquet (Spark-readable dataset)


Language
Python 3.8+


Project Structure
FinalFacialFunctions/
├── facialRecognition/          # Django app
├── FinalFacialFunctions/       # Django settings
├── media/uploads/              # Uploaded images storage
├── saved_model/
│   ├── my_trained_modelH5.h5   # EfficientNetB0 model
│   ├── saved_lsh_modelv2       # Spark LSH model (saved)
│   └── df_facesv2.parquet      # Spark DataFrame with embeddings
├── spark-3.2.1-bin-hadoop3.2/  # Spark (local)
├── hadoop-3.2.1/               # Hadoop binaries
├── manage.py
└── requirements.txt

How to Run Locally
1. Install Dependencies
Create a virtual environment and install dependencies:
pip install -r requirements.txt

Requirements:

Python 3.8–3.10
Java 8 or 11 (Note: Java 17/21/24 is incompatible with Spark 3.2.1)
Spark 3.2.1 + Hadoop 3.2 (downloaded locally)

2. Set Environment Variables (PowerShell)
$env:JAVA_HOME="C:\Program Files\Eclipse Adoptium\jdk-11.0.27.6-hotspot"
$env:SPARK_HOME="D:\FinalFacialFunctions\spark-3.2.1-bin-hadoop3.2"
$env:HADOOP_HOME="D:\FinalFacialFunctions\hadoop-3.2.1"
$env:PATH="$env:JAVA_HOME\bin;$env:SPARK_HOME\bin;$env:HADOOP_HOME\bin;$env:PATH"

Replace paths with your actual directory locations as needed.
3. Run the Django Server
python manage.py runserver 0.0.0.0:8000

Access the application at:http://localhost:8000/
