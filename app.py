import os
import re
import PyPDF2
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

REQUIRED_SKILLS = [
    "python", "java", "spring boot", "sql", "machine learning", "ml", 
    "deep learning", "react", "node.js", "javascript", "c++", "backend", "frontend"
]

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file and normalize it."""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + " "
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    
    return text.lower().strip()

def extract_contact_and_year(text):
    """Extract contact number and graduation year from resume text."""
    contact_match = re.search(r"\b\d{10}\b", text)
    year_match = re.search(r"\b(20\d{2})\b", text)

    contact = contact_match.group() if contact_match else "N/A"
    year = year_match.group() if year_match else "Unknown"
    
    return contact, year

@app.route('/upload_file', methods=['POST'])
def upload_file():
    """Filter candidates based on skill match and recruiter preference for number of selections."""
    
    data = request.get_json()
    top_n = data.get("top_n", 5)  # Default to 5 if not provided

    candidate_data = []
    
    # Process all resumes in the uploads folder
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)

        resume_text = extract_text_from_pdf(file_path)
        contact, year = extract_contact_and_year(resume_text)

        candidate_data.append({
            "name": filename.replace(".pdf", ""),
            "text": resume_text,
            "contact": contact,

        })

    # Convert skills and resumes into vectors using TF-IDF
    all_texts = [data["text"] for data in candidate_data]
    tfidf_vectorizer = TfidfVectorizer(vocabulary=REQUIRED_SKILLS, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)

    skills_vector = tfidf_vectorizer.transform([" ".join(REQUIRED_SKILLS)]).toarray()[0]
    if skills_vector.sum() == 0:
        return jsonify({"error": "No matching skills found in resumes"}), 400

    scores = cosine_similarity(tfidf_matrix, [skills_vector])

    for i, data in enumerate(candidate_data):
        data["score"] = round(float(scores[i][0]), 4)

    # Sort candidates by score in descending order
    candidate_data = sorted(candidate_data, key=lambda x: x["score"], reverse=True)

    # Get top N candidates
    selected_candidates = candidate_data[:top_n]

    return jsonify({"selected_candidates": selected_candidates, "status": "success"}), 200

@app.route('/', methods=['GET'])
def welcomePage():
    test = "The server is up and running"
    return test

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Get port from environment variable or default to 5000
    app.run(host="0.0.0.0", port=port, debug=True)