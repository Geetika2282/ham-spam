import gradio as gr
import joblib

model = joblib.load("spam_classification.pkl")

def predict(text):
    return model.predict([text])[0]

gr.interface(fn=predict, inputs="text", outputs="text").launch()