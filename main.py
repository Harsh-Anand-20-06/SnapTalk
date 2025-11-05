import streamlit as st
from PIL import Image
import pytesseract
import re
import spacy
from transformers import pipeline
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from processor import TextProcessor

st.set_page_config(page_title="AI Image Text Analyzer", layout="wide")

nlp = spacy.load("en_core_web_sm")
sentiment_analyzer = pipeline("sentiment-analysis")
summarizer = pipeline("summarization")

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task = 'text_generation',
    provider = 'auto'
)

chat_model = ChatHuggingFace(llm=llm)

st.title("AI Image Text Analyzer")
st.write("Upload an image with text — the app will extract, clean, analyze, and even summarize it!")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.read())


    processor = TextProcessor("temp_image.png")

    st.subheader("Extracted Text")
    st.write(processor.text or "_No text found._")


    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Tokens",
        "Named Entities",
        "Sentiment",
        "Summary",
        "Ask AI",
        "Keywords",
        "Chat about Extracted text"
    ])

    with tab1:
        tokens = processor.get_tokens()
        st.write(tokens)

    with tab2:
        entities = processor.get_entities()
        if entities:
            for ent, label in entities:
                st.markdown(f"**{ent}** — `{label}`")
        else:
            st.write("No named entities found.")

    with tab3:
        sentiment = processor.analyze_sentiment()
        st.json(sentiment)

    with tab4:
        st.write("**Summary:**")
        st.write(processor.summarize_text())

    with tab5:
        st.subheader("Ask AI about the extracted text")

        if st.button("Ask AI"):
            with st.spinner("Thinking..."):
                response = processor.ask_question()
            st.write(response)

    with tab6:
        st.subheader("Top Keywords from Extracted Text")
        if processor.text.strip() == "":
            st.warning("Please upload an image first to extract text.")
        else:
            top_n = st.slider("Select number of keywords", 3, 15, 5)
            keywords = processor.get_keywords(top_n=top_n)
            if keywords:
                st.success(" Extracted Keywords:")
                st.write(", ".join(keywords))
            else:
                st.info("No keywords found — try another image.")

    with tab7:
        st.subheader("Chat about Extracted Text ")
        if processor.text.strip() == "":
            st.warning("Please upload an image first to extract text.")
        else:
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            user_input = st.text_input("Ask a question about the extracted text:")
            if st.button("Send"):
                if user_input.strip() == "":
                    st.warning("Please type a question.")
                else:
                    answer = processor.ask_questions_about_text(user_input)
                    st.session_state.chat_history.append(("You", user_input))
                    st.session_state.chat_history.append(("Bot", answer))

            if st.session_state.chat_history:
                st.markdown("### Conversation:")
                for role, message in st.session_state.chat_history:
                    if role == "You":
                        st.markdown(f" **{role}:** {message}")
                    else:
                        st.markdown(f" **{role}:** {message}")

else:
    st.info("Upload an image to start analyzing!")


st.markdown("---")
st.caption("Built using Streamlit, spaCy, Transformers, and LangChain.")
