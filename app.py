# app.py

import gradio as gr
import joblib

model = joblib.load("spam_classifier.pkl")

def predict_spam(text):
    pred = model.predict([text])[0]
    return "Spam" if pred == 1 else "Ham"

gr.Interface(
    fn=predict_spam,
    inputs=gr.Textbox(label="Enter SMS Text"),
    outputs=gr.Textbox(label="Prediction"),
    title="SMS Spam Classifier",
    description="Enter an SMS message to check if it's Spam or Ham."
).launch()
