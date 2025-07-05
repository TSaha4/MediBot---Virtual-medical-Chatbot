from flask import Flask, request, jsonify
import pandas as pd
from dotenv import load_dotenv
import os
import ibm_boto3
from ibm_botocore.client import Config
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import io
import json
from fuzzywuzzy import process, fuzz
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# COS credentials
load_dotenv()
cos = ibm_boto3.client(
    service_name='s3',
    ibm_api_key_id=os.environ.get('IBM_API_KEY'),
    ibm_auth_endpoint='https://iam.cloud.ibm.com/identity/token',
    config=Config(signature_version='oauth'),
    endpoint_url=os.environ.get('IBM_COS_ENDPOINT')
)

bucket = os.environ.get('IBM_COS_BUCKET')
symptoms_file = 'Symptom-severity.csv'
disease_file = 'dataset.csv'
dept_file = 'disease-department dataset.csv'
synonym_file = 'keywords.json'  # New JSON file for medical keywords

def load_kb():
    # Load symptom-severity
    body = cos.get_object(Bucket=bucket, Key=symptoms_file)['Body'].read()
    symptoms_df = pd.read_csv(io.BytesIO(body))
        
    # Load disease-symptom (wide format)
    body2 = cos.get_object(Bucket=bucket, Key=disease_file)['Body'].read()
    disease_df = pd.read_csv(io.BytesIO(body2))
        
    # Load disease-department mapping
    body3 = cos.get_object(Bucket=bucket, Key=dept_file)['Body'].read()
    dept_df = pd.read_csv(io.BytesIO(body3))
        
    # Load medical synonym mapping from JSON
    body4 = cos.get_object(Bucket=bucket, Key=synonym_file)['Body'].read()
    medical_keywords = json.loads(body4.decode('utf-8'))
        
    return symptoms_df, disease_df, dept_df, medical_keywords

# Load datasets
symptoms_df, disease_df, dept_df, medical_keywords = load_kb()

# --- ENHANCED RAG PREPROCESSING WITH ERROR FIXES ---

# Clean and standardize column names
symptoms_df.columns = symptoms_df.columns.str.strip()
disease_df.columns = disease_df.columns.str.strip()
dept_df.columns = dept_df.columns.str.strip()

# Handle case sensitivity and clean symptom names
symptoms_df['symptom'] = symptoms_df['symptom'].str.strip()
all_symptoms = [s.strip() for s in symptoms_df['symptom'].tolist()]

# 1. Reshape disease-symptom dataset from wide to long format (CORRECTED)
# Find symptom columns correctly
symptom_cols = [col for col in disease_df.columns if col.lower().startswith('symptom')]
if not symptom_cols:
    # Fallback: assume all columns except first are symptoms
    symptom_cols = disease_df.columns[1:].tolist()

# Use correct column names for melting
disease_col = disease_df.columns[0]  # First column should be disease
long_df = disease_df.melt(id_vars=[disease_col], value_vars=symptom_cols, value_name='symptom')
long_df = long_df.dropna(subset=['symptom'])
long_df = long_df.rename(columns={disease_col: 'disease'})

# Clean the symptom data in long_df
long_df['symptom'] = long_df['symptom'].str.strip()

# 2. Enhanced symptom processing
symptom_aliases = [s.replace('_', ' ') for s in all_symptoms]

# 3. Multi-model approach for better accuracy
model = SentenceTransformer('all-MiniLM-L6-v2')
symptom_embeddings = model.encode(all_symptoms)
index = faiss.IndexFlatL2(symptom_embeddings.shape[1])
index.add(np.array(symptom_embeddings))

# TF-IDF for lexical matching
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(symptom_aliases)

# 4. Department mapping dictionary (with error handling)
try:
    disease_to_department = dict(zip(dept_df.iloc[:, 0], dept_df.iloc[:, 1]))
except Exception as e:
    print(f"Error creating department mapping: {e}")
    disease_to_department = {}

def get_department(likely_disease, mapping, threshold=75):
    if not mapping:
        return "General Medicine"
    
    choices = list(mapping.keys())
    if not choices:
        return "General Medicine"
    
    best_match, score = process.extractOne(likely_disease, choices)
    if score >= threshold:
        return mapping[best_match]
    else:
        return "General Medicine"

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_symptoms_advanced(user_input, symptom_list, symptom_aliases, medical_keywords, threshold_fuzzy=70, threshold_semantic=0.4):
    user_input_clean = preprocess_text(user_input)
    found_symptoms = set()
    
    # Stage 1: Medical keyword mapping from JSON
    for keyword, mapped_symptoms in medical_keywords.items():
        if keyword.lower() in user_input_clean:
            for symptom in mapped_symptoms:
                # Clean symptom names to match dataset format
                clean_symptom = symptom.strip()
                if clean_symptom in symptom_list:
                    found_symptoms.add(clean_symptom)
    
    # Stage 2: N-gram fuzzy matching
    tokens = user_input_clean.split()
    for n in range(1, 4):  # 1-3 word combinations
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i+n])
            
            # Fuzzy match against symptom aliases
            if symptom_aliases:
                best_match, score = process.extractOne(ngram, symptom_aliases)
                if score >= threshold_fuzzy:
                    try:
                        idx = symptom_aliases.index(best_match)
                        found_symptoms.add(symptom_list[idx])
                    except (ValueError, IndexError):
                        continue
    
    # Stage 3: Semantic similarity using embeddings
    try:
        user_emb = model.encode([user_input_clean])
        D, I = index.search(np.array(user_emb), k=5)  # Top 5 matches
        
        for i, (distance, idx) in enumerate(zip(D[0], I[0])):
            similarity = 1 / (1 + distance)  # Convert distance to similarity
            if similarity >= threshold_semantic:
                if 0 <= idx < len(symptom_list):
                    found_symptoms.add(symptom_list[idx])
    except Exception as e:
        print(f"Embedding search error: {e}")
    
    # Stage 4: TF-IDF lexical matching
    try:
        user_tfidf = tfidf_vectorizer.transform([user_input_clean])
        tfidf_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
        
        top_tfidf_indices = np.argsort(tfidf_similarities)[-3:]  # Top 3
        for idx in top_tfidf_indices:
            if tfidf_similarities[idx] >= 0.1 and 0 <= idx < len(symptom_list):
                found_symptoms.add(symptom_list[idx])
    except Exception as e:
        print(f"TF-IDF matching error: {e}")
    
    return list(found_symptoms)

def weighted_disease_scoring(matched_symptoms, long_df, symptom_severity_df):
    # Enhanced disease scoring using weighted approach with error handling
    if not matched_symptoms:
        return "General Condition", 0
    
    # Get diseases that match symptoms
    matches = long_df[long_df['symptom'].isin(matched_symptoms)]
    
    if matches.empty:
        return "General Condition", 0
    
    # Calculate weighted scores based on symptom severity
    disease_scores = {}
    for disease in matches['disease'].unique():
        disease_symptoms = matches[matches['disease'] == disease]['symptom'].tolist()
        
        # Count matching symptoms
        match_count = len([s for s in matched_symptoms if s in disease_symptoms])
        total_disease_symptoms = len(disease_symptoms)
        
        # Calculate coverage score
        coverage_score = match_count / max(total_disease_symptoms, 1)
        
        # Add severity weighting
        severity_weight = 0
        for symptom in matched_symptoms:
            if symptom in disease_symptoms:
                severity_row = symptom_severity_df[symptom_severity_df['symptom'] == symptom]
                if not severity_row.empty:
                    severity_weight += severity_row['severity'].iloc[0]
        
        # Combined score with improved weighting
        final_score = (match_count * 3) + (coverage_score * 2) + (severity_weight * 0.2)
        disease_scores[disease] = final_score
    
    if disease_scores:
        best_disease = max(disease_scores, key=disease_scores.get)
        return best_disease, disease_scores[best_disease]
    
    return "General Condition", 0

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        user_symptoms = data.get('symptoms', '')
        duration = data.get('duration', '')
        severity = data.get('severity', '')
        history = data.get('history', '')

        # --- ENHANCED MULTI-STAGE SYMPTOM EXTRACTION ---
        matched_symptoms = extract_symptoms_advanced(
            user_symptoms, 
            all_symptoms, 
            symptom_aliases,
            medical_keywords,
            threshold_fuzzy=70,
            threshold_semantic=0.4
        )

        if not matched_symptoms:
            return jsonify({
                'department': "General Medicine",
                'disease': "General Condition", 
                'severity_score': 0,
                'matched_symptoms': [],
                'confidence': 'low',
                'context': f"No specific symptoms identified from: {user_symptoms}"
            })

        # --- ENHANCED DISEASE PREDICTION ---
        likely_disease, disease_score = weighted_disease_scoring(
            matched_symptoms, long_df, symptoms_df
        )

        # Calculate total severity
        total_severity = symptoms_df[symptoms_df['symptom'].isin(matched_symptoms)]['severity'].sum()

        # Enhanced department mapping
        department = get_department(likely_disease, disease_to_department, threshold=70)

        # Confidence calculation based on multiple factors
        confidence = 'high' if disease_score > 8 and len(matched_symptoms) >= 3 else \
                    'medium' if disease_score > 4 and len(matched_symptoms) >= 2 else 'low'

        # Context for debugging
        context = f"Matched symptoms: {matched_symptoms}. Disease: {likely_disease} (score: {disease_score:.2f}). Keywords used: {len([k for k in medical_keywords.keys() if k in user_symptoms.lower()])} from JSON"

        return jsonify({
            'department': department,
            'disease': likely_disease,
            'severity_score': int(total_severity),
            'matched_symptoms': matched_symptoms,
            'confidence': confidence,
            'context': context
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    # Health check endpoint
    return jsonify({
        "status": "healthy",
        "symptoms_loaded": len(all_symptoms),
        "diseases_loaded": len(long_df['disease'].unique()) if not long_df.empty else 0,
        "departments_loaded": len(disease_to_department),
        "medical_keywords_loaded": len(medical_keywords)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
