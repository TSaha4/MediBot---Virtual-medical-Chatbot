# MediBot---Virtual-medical-Chatbot

An AI-powered medical triage chatbot that interprets user-described symptoms and recommends the most appropriate medical department or specialist. Built with Python Flask, IBM Watson Assistant, and cloud-based data integration.

**Features**
- Conversational AI: Natural language symptom collection and clarification via Watson Assistant.
- Advanced Symptom Extraction: Multi-stage pipeline including keyword mapping, fuzzy matching, semantic similarity (SentenceTransformers + FAISS), and TF-IDF lexical matching.
- Disease Prediction & Triage: Weighted scoring based on matched symptoms, coverage, and severity for accurate department recommendations.
- Cloud Integration: Secure data storage and retrieval using IBM Cloud Object Storage (COS).
- Real-Time Web Chat: Seamless user interaction via an embedded Watson Assistant widget.
- Extensible & Cloud-Ready: Modular backend, OpenAPI integration, and environment variable-based configuration.
  
**Architecture**
- Frontend: Custom HTML/CSS website with embedded Watson Assistant web chat widget.
- Backend: Python Flask REST API for symptom analysis (/analyze) and health checks (/health).
- AI/NLP: Symptom extraction pipeline using pandas, fuzzywuzzy, SentenceTransformers, FAISS, and scikit-learn.
- Data: Medical datasets (symptom-severity, disease-symptom, disease-department, synonym mapping) loaded from IBM COS.
- Integration: Watson Assistant custom extension (OpenAPI/Swagger) connects the dialog flow to the backend API via ngrok.

**Flowchart**

<img width="706" height="1491" alt="image" src="https://github.com/user-attachments/assets/2434896b-ca28-4746-b932-ba331adf341e" />


**Setup & Deployment**

**Prerequisites:**
- Python 3.8 or above
- IBM Cloud Object Storage account
- IBM Watson Assistant instance
- ngrok account (for static public URL)

**Required Python packages:**
pip install -r requirements.txt

**Environment Variables:**
Create a `.env` file with the following keys:
IBM_API_KEY=your_ibm_api_key
IBM_COS_ENDPOINT=your_cos_endpoint
IBM_COS_BUCKET=your_cos_bucket
PORT=5000

**Running the Backend:**
python app.py

**Exposing the Backend:**
ngrok http 5000
Use the generated ngrok URL for Watson Assistant integration and OpenAPI.
Setting Up Watson Assistant
1. Create a new Assistant in Watson Assistant.
2. Configure dialog nodes for symptom collection.
3. Go to Integrations > Extensions > Build custom extension.
4. Import the provided OpenAPI (Swagger) JSON for the `/analyze` endpoint.
5. Map dialog variables to extension parameters.
6. Embed the Watson Assistant web chat widget in your `index.html`.

**API Reference**
/analyze (POST)

**Request:**
{
  "symptoms": "string",
  "duration": "string",
  "severity": "string",
  "history": "string"
}

**Response:**
{
  "department": "string",
  "disease": "string",
  "severity_score": 0
}

/health (GET)
Returns system status and loaded data counts.

**Testing & Validation**
- Unit and integration tests implemented for core logic and NLP pipeline.
- `/health` endpoint helps with backend monitoring.
- Iterative improvements made based on real user feedback.


**Screen Shots**

<img width="1253" height="669" alt="image" src="https://github.com/user-attachments/assets/1f2bad6c-6737-44dd-98bd-f16a5c96fd3c" />
<img width="1253" height="658" alt="image" src="https://github.com/user-attachments/assets/83749fa7-26de-4ae6-9b31-d8b9fcc22968" />


1st Case (Disease: Tuberculosis):

<img width="655" height="1084" alt="image" src="https://github.com/user-attachments/assets/c01e25b0-c121-40b2-88df-a21ee0c3c40f" />
<img width="640" height="1075" alt="image" src="https://github.com/user-attachments/assets/e0f1f083-3580-4a9f-ab8a-396b39e59a46" />
<img width="640" height="1077" alt="image" src="https://github.com/user-attachments/assets/69c5b1c1-a32f-4b71-8d97-e5858c1094dd" />
<img width="646" height="1081" alt="image" src="https://github.com/user-attachments/assets/fba41f22-8716-42c3-8c2a-07c9150b5164" />

Result:

<img width="645" height="517" alt="image" src="https://github.com/user-attachments/assets/d3e5a014-70a8-48f1-bce6-135bb368c1e2" />


2nd Case (Disease: Diabetes):

<img width="651" height="814" alt="image" src="https://github.com/user-attachments/assets/a80bc11f-9e0c-45e7-b544-768bdabaf663" />
<img width="643" height="799" alt="image" src="https://github.com/user-attachments/assets/3bc80910-4857-46a8-9cad-f795079ca4bd" />
<img width="649" height="910" alt="image" src="https://github.com/user-attachments/assets/c87b086f-93c5-483f-bee9-778257fbfd2c" />
<img width="645" height="874" alt="image" src="https://github.com/user-attachments/assets/db450bf6-80d2-4d77-8d5c-f66f9d9aa70f" />

Result:

<img width="637" height="438" alt="image" src="https://github.com/user-attachments/assets/ea4a42aa-75d7-4935-b74d-22768966c78c" />

