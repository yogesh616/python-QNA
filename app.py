from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import re
 
app = Flask(__name__)

# Initialize question-answering pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
#qa_pipeline = pipeline("question-answering", model="bert-base-multilingual-cased")
def preprocess_text(text):
    # Remove extra spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters (optional)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Convert text to lowercase
    text = text.lower()
    
    return text

@app.route('/summarize', methods=['POST'])
def answer_question():
    data = request.json
    question = data.get('question', '')
    context = data.get('context', '')

    if not question or not context:
        return jsonify({"error": "Both question and context are required"}), 400

    # Preprocess context and question
    processed_context = preprocess_text(context)
    processed_question = preprocess_text(question)

    # Get the answer
    result = qa_pipeline(question=processed_question, context=processed_context)
    answer = result['answer']
    
    return jsonify({"summary": answer})

@app.route('/')
def home():
    return render_template('index.html')