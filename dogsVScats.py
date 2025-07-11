import streamlit as st
import os

from langchain_groq import ChatGroq
from langchain.agents import initialize_agent
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import StructuredTool
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

st.title("Cat vs Dog Conversational Classifier")

api_key = st.text_input("Enter your Groq API Key", type="password")

if api_key:
    os.environ["GROQ_API_KEY"] = api_key

    model = load_model('cats_vs_dogs_model.h5')

    def classify_image(img_path: str) -> str:
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)

        if prediction[0][0] > 0.5:
            return "DOG"
        else:
            return "CAT"

    tool = StructuredTool.from_function(
        classify_image,
        name="classify_image",
        description="Classify an image as a cat or dog."
    )

    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.3
    )

    agent = initialize_agent(
        tools=[tool],
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True
    )

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        with open("temp.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        user_query = "Classify this image: temp.jpg"
        classification = agent.run(user_query)

        st.write(f"Prediction: {classification}")

        prompt = PromptTemplate(
            input_variables=["result"],
            template="You are an animal expert. Based on the image classification result, {result}, provide a brief description of the animal, its possible breed, personality, and care tips."
        )

        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt
        )

        description = llm_chain.run({"result": classification})
        st.write(description)
