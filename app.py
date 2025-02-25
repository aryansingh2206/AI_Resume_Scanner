from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
import pdfplumber
import re
import nltk
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
import numpy as np
import pickle
import os

# Download NLTK stopwords if not already available
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

# Load Pre-trained Word2Vec Model
word2vec_model_path = "GoogleNews-vectors-negative300.bin"
if os.path.exists(word2vec_model_path):
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)
    print("Word2Vec model loaded successfully.")
else:
    raise FileNotFoundError("Error: Pre-trained Word2Vec model not found!")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Connect to PostgreSQL
DB_NAME = "resume_scanner"
DB_USER = "aryan"
DB_PASS = "Kickstart@2010"
DB_HOST = "localhost"
DB_PORT = "5432"

try:
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        host=DB_HOST,
        port=DB_PORT,
    )
    cursor = conn.cursor()
    print("Database connected successfully.")
except Exception as e:
    raise ConnectionError(f"Database connection failed: {e}")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        return text.strip() if text else ""
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

# Preprocess text
def clean_text(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return [word for word in words if word not in stop_words]

# Function to calculate Word2Vec similarity
def word2vec_similarity(resume_text, job_desc):
    resume_words = clean_text(resume_text)
    job_words = clean_text(job_desc)
    
    resume_vectors = [word2vec_model[word] for word in resume_words if word in word2vec_model]
    job_vectors = [word2vec_model[word] for word in job_words if word in word2vec_model]
    
    if not resume_vectors or not job_vectors:
        return 0  # Return 0% similarity if no common words exist

    resume_vector = np.mean(resume_vectors, axis=0)
    job_vector = np.mean(job_vectors, axis=0)
    
    similarity = np.dot(resume_vector, job_vector) / (np.linalg.norm(resume_vector) * np.linalg.norm(job_vector))
    return round(similarity * 100, 2)

# Function to predict match using Logistic Regression
def predict_match(resume_text, job_desc):
    model_path = "resume_classifier.pkl"
    vectorizer_path = "vectorizer.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("Warning: Model files not found. Defaulting to 'No Match'.")
        return 0  # Return No Match if model files are missing
    
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    with open(vectorizer_path, "rb") as vec_file:
        vectorizer = pickle.load(vec_file)
    
    resume_text_clean = " ".join(clean_text(resume_text))
    job_desc_clean = " ".join(clean_text(job_desc))
    
    X_new = vectorizer.transform([resume_text_clean + " " + job_desc_clean])
    prediction = model.predict(X_new)[0]
    
    print(f"DEBUG: Model prediction: {prediction}")  # Debugging Output
    return int(prediction)

# API route to analyze resumes
@app.route("/analyze", methods=["POST"])
def analyze_resume():
    if "resume" not in request.files or "job_description" not in request.form:
        return jsonify({"error": "Missing resume file or job description"}), 400
    
    file = request.files["resume"]
    job_desc = request.form["job_description"].strip()
    
    if file.filename == "":
        return jsonify({"error": "Empty file uploaded"}), 400
    
    try:
        resume_text = extract_text_from_pdf(file)
        if not resume_text:
            return jsonify({"error": "Failed to extract text from resume"}), 400
        
        match_prediction = predict_match(resume_text, job_desc)
        match_percentage = word2vec_similarity(resume_text, job_desc)
        
        cursor.execute(
            "INSERT INTO resumes (name, content, score) VALUES (%s, %s, %s)",
            (file.filename, resume_text, match_percentage),
        )
        conn.commit()
        
        return jsonify({
            "classification": "Good Match" if match_prediction == 1 else "No Match",
            "matching_percentage": match_percentage
        })
    except Exception as e:
        print(f"Error processing resume: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True)
