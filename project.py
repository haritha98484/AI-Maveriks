import faiss
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
def load_document(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.readlines()
questions = load_document("questions.txt")  # List of questions
answers_doc = load_document("answers.txt")  # Large document with answers
# Load Sentence Transformer Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert answers into embeddings for search
answer_embeddings = embedding_model.encode(answers_doc)

# Index with FAISS for fast search
index = faiss.IndexFlatL2(answer_embeddings.shape[1])
index.add(np.array(answer_embeddings).astype("float32"))
def find_best_answer(question):
    question_embedding = embedding_model.encode([question])
    _, idx = index.search(np.array(question_embedding).astype("float32"), 1)
    return answers_doc[idx[0][0]]

# Test on a sample question
sample_question = "What is AI alignment?"
best_answer = find_best_answer(sample_question)
print(f"Best Answer: {best_answer}")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

def validate_answer(question, answer):
    response = qa_pipeline(question=question, context=answer)
    return response["answer"], response["score"]

# Test Validation
validated_answer, confidence = validate_answer(sample_question, best_answer)
print(f"Validated Answer: {validated_answer} | Confidence: {confidence}")
updated_questions = []

for question in questions:
    best_answer = find_best_answer(question)
    validated_answer, confidence = validate_answer(question, best_answer)

    if confidence > 0.7:  # Confidence threshold
        label = "True"
    else:
        label = "False"

    updated_questions.append(f"{question.strip()} - {label} (Reason: {validated_answer})\n")

# Save Updated Questions
with open("updated_questions.txt", "w", encoding="utf-8") as file:
    file.writelines(updated_questions)
