import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from utils.preprocess import preprocess_text

# Load preprocessed sample data
data = pd.DataFrame([
    {"text": "machine learning python data analysis", "job_title": "Data Scientist"},
    {"text": "java spring backend microservices", "job_title": "Backend Developer"},
    {"text": "marketing seo content strategy", "job_title": "Digital Marketer"}
])

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(data['text']).toarray()

# Clustering resumes
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['cluster'] = kmeans.fit_predict(X)

# Train classifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X, data['job_title'])

def predict_candidates(resumes, required_skills):
    """
    Select top resumes based on recruiter-specified skills.
    """
    # Convert required skills to TF-IDF features
    required_skills_vector = vectorizer.transform([required_skills]).toarray()

    ranked_resumes = []
    for resume in resumes:
        resume_vector = vectorizer.transform([" ".join(resume["text"])]).toarray()
        similarity = cosine_similarity(resume_vector, required_skills_vector)[0][0]
        ranked_resumes.append((resume, similarity))

    # Sort by similarity score
    ranked_resumes = sorted(ranked_resumes, key=lambda x: x[1], reverse=True)

    # Select top candidates
    selected_candidates = [
        {"name": res[0]["name"], "year": res[0]["year"], "contact": res[0]["contact"], "score": res[1]}
        for res in ranked_resumes
    ]

    return selected_candidates
