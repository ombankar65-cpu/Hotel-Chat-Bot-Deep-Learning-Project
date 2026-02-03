import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load files
model = load_model("chatbot_model.h5")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

data = pd.read_csv("data.csv")
menu = pd.read_csv("menu.csv")

st.title("🏨 Hotel Ordering Chatbot")

user_input = st.text_input("You:")

CONFIDENCE_THRESHOLD = 0.70

def show_menu():
    st.table(menu[["dish", "price"]])

if user_input:
    seq = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(seq, maxlen=model.input_shape[1], padding="post")
    prediction = model.predict(padded)
    confidence = np.max(prediction)
    intent = label_encoder.inverse_transform([np.argmax(prediction)])[0]

    if confidence < CONFIDENCE_THRESHOLD:
        st.warning("❌ Sorry, I didn't understand. Please try again.")
    
    elif intent == "menu":
        st.success("🍽️ Our Menu")
        show_menu()

    elif intent == "order":
        st.info("✍️ Please type the dish name to order")

    else:
        response = data[data["intent"] == intent]["response"].iloc[0]
        st.success(f"Bot: {response}")

    # Order logic
    for dish in menu["dish"]:
        if dish.lower() in user_input.lower():
            dish_data = menu[menu["dish"] == dish].iloc[0]
            if dish_data["available"] == "yes":
                st.success(f"✅ {dish} ordered successfully! Price: ₹{dish_data['price']}")
            else:
                st.error(f"❌ Sorry, {dish} is currently not available.")
