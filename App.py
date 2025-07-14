import streamlit as st
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime
import io
import xlrd
import xlwt
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Sentiment Analysis App", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

@st.cache_resource
def load_model_and_tokenizer(model_path="My_Model"):
  tokenizer = AutoTokenizer.from_pretrained(model_path)
  model = AutoModelForSequenceClassification.from_pretrained(model_path)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  return tokenizer, model, device

tokenizer, model, device = load_model_and_tokenizer()

def predict_sentiment(text):
    no_match_phrases = ["nothing", "no coments", "no suggestions", "no thanks", "none", "nil", "n/a",]
    if not text or text.strip() == "" or text.lower().strip() in no_match_phrases:
        return "No-Match"

    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    
    with torch.no_grad():
        outputs = model(**tokens)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(scores).item()

    labels = ["Positive", "Negative", "Neutral", "No-Match"]
    return labels[predicted_class] if predicted_class < len(labels) else "Unknown"

st.title(" 👥Employee's Sentiment Analysis Chatbot")
st.markdown("W E L C O M E 💛 ")
st.markdown("Let's chat with model to get sentiment prediction on your feedback.")

with st.chat_message("user"):
    user_input = st.text_input("Enter your Feedback", key="feedback_input")
    submit = st.button("Submit")

if submit and user_input:
    already_exists = any(entry["Feedback"] == user_input for entry in st.session_state.chat_history)
    if not already_exists:
        sentiment = predict_sentiment(user_input)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        emoji_map = {"Positive": "😊", "Negative": "😞", "Neutral": "😐", "No-Match": "❓"}

        st.session_state.chat_history.append({
            "timestamp": timestamp,
            "Feedback": user_input,
            "Sentiment": sentiment,
        })

        with st.chat_message("assistant"):
            st.markdown(f"**Sentiment:** {sentiment} {emoji_map.get(sentiment, '')}")
    else:
        st.warning("This feedback has already been submitted.")

if st.session_state.chat_history:
  st.subheader("Prediction History")
  history_df = pd.DataFrame(st.session_state.chat_history)
  st.dataframe(history_df, use_container_width=True)
  

  output = io.BytesIO()
  workbook = xlwt.Workbook()
  sheet = workbook.add_sheet("Prediction")
  for col_idx,col in enumerate(history_df.columns):
    sheet.write(0, col_idx, col)
  for row_idx, row in enumerate(history_df.itertuples(index=False), start=1):
    for col_idx, value in enumerate(row):
      sheet.write(row_idx, col_idx, value)
  workbook.save(output)
  st.download_button(label="Download History as Excel", data=output.getvalue(), file_name="prediction_history.xls")

  st.markdown(" Thank you! Made by Nidhi Sahani 💛 ")
