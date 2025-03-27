from flask import Flask, request, render_template
import os
import pdfplumber
import docx2txt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Job description
job_description = """
3+ years of experience in implementing and deploying Django applications in an enterprise-grade environment.
Expertise in using version control systems like Git, GitHub, and GitLab.
Experience with Python frameworks like Django and FastAPI.
Experience in creating and scaling highly performing REST APIs.
Experience working with PostgreSQL, and MySQL.
Familiarity with cloud platforms such as AWS, Azure, etc.
Must have a hands-on experience with the OpenAI model (such as GPT4 / 4o mini)
Should know LLM Model Training and Fine-tuning.
Should know OpenCV and Pytorch / TensorFlow.
Good to have:
Experience in generative AI projects in real-world applications.
Knowledge of the langchain package and its architecture.
Familiarity with video processing techniques and tools.
Understanding of LLM architecture and its use cases.
Experience optimizing and fine-tuning generative models to improve performance, efficiency, and scalability.
Knowledge of any No SQL database.
Knowledge of Chroma DB or Pinecone Vector Db
Design, develop, and refine AI-generated text prompts for various applications.
"""

# Function to extract text from docx files
def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

# Function to extract text from pdf files
def extract_text_from_pdf(file_path):
    text = ''
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + '\n'
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'resume' not in request.files:
        return "No file uploaded"
    file = request.files['resume']
    if file.filename == '':
        return "No selected file"
    
    file_extension = os.path.splitext(file.filename)[-1].lower()
    text = ''
    
    if file_extension == '.pdf':
        try:
            text = extract_text_from_pdf(file)
        except Exception as e:
            return f"Error processing PDF: {str(e)}"
    elif file_extension == '.docx':
        try:
            text = extract_text_from_docx(file)
        except Exception as e:
            return f"Error processing DOCX: {str(e)}"
    else:
        return "Unsupported file format. Please upload a PDF or DOCX file."
    
    # Compute similarity
    vectorizer = TfidfVectorizer()
    corpus = [job_description.lower(), text.lower()]
    tfidf_matrix = vectorizer.fit_transform(corpus)
    cosine_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()[0]
    score = round(1 + (cosine_score * 9))
    
    return render_template('result.html', score=score)

if __name__ == '__main__':
    app.run(debug=True)
