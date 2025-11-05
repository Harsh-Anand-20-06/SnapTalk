import os
from api_key import key
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import spacy
from transformers import pipeline
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")



nlp = spacy.load("en_core_web_sm")
sentiment_analyzer = pipeline("sentiment-analysis")
summarizer = pipeline("summarization")


os.environ['HUGGINGFACEHUB_API_TOKEN'] = key

from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task = 'text_generation',
    provider = 'auto'
)

chat_model = ChatHuggingFace(llm=llm)
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

from langchain.agents import create_agent
agent = create_agent(chat_model,[wiki],checkpointer = InMemorySaver())




class TextProcessor:
    def __init__(self, image_path):
        self.img_path = image_path
        self.text = self.show_text()

    def show_text(self):
        img = Image.open(self.img_path)
        raw_text = pytesseract.image_to_string(img)

        cleaned_text = re.sub(r'\s+', ' ', raw_text)
        cleaned_text = re.sub(r'[^\x00-\x7F]+', '', cleaned_text)
        cleaned_text = cleaned_text.strip()
        return cleaned_text

    def analyze_sentiment(self):
        return sentiment_analyzer(self.text)

    def get_tokens(self):
        doc = nlp(self.text)
        return [token.text for token in doc if token.is_alpha or token.is_digit or token.is_punct]

    def get_entities(self):
        doc = nlp(self.text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def summarize_text(self):
        if len(self.text.split()) < 50:
            return "Text too short to summarize."
        summary = summarizer(self.text, max_length=100, min_length=30, do_sample=False)
        return summary[0]['summary_text']

    def ask_question(self):
        response = chat_model.invoke(self.text)
        res = response.content
        clean_response = re.sub(r"<think>.*?</think>", "", res, flags=re.DOTALL).strip()
        cleaned_text = re.sub(r'\s+', ' ', clean_response)
        cleaned_text = re.sub(r'[^\x00-\x7F]+', '', cleaned_text)
        cleaned_text = cleaned_text.strip()
        cleaned_text = cleaned_text.replace("**", "")
        cleaned_text = cleaned_text.replace("*", "")
        return cleaned_text

    def ask_questions_about_text(self, user_question):
        if len(self.text.strip()) == 0:
            return "No text found in the image."
        input_ = agent.invoke(
            {"messages": [{"role": "user", "content": self.text}]},
            {"configurable": {"thread_id": "1"}},
        )
        output_ = agent.invoke(
            {"messages": [{'role': 'user', 'content': user_question}]},
            {"configurable": {'thread_id': '1'}},
        )
        if isinstance(output_, dict) and "messages" in output_:
            raw_text = output_["messages"][-1].content
        elif hasattr(output_, "content"):  # for direct AIMessage
            raw_text = output_.content
        else:
            raw_text = str(output_)

        cleaned_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = re.sub(r'[^\x00-\x7F]+', '', cleaned_text)
        cleaned_text = cleaned_text.strip()
        return cleaned_text

    def get_keywords(self, top_n=5):
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform([self.text])
        scores = zip(vectorizer.get_feature_names_out(), X.toarray()[0])
        sorted_words = sorted(scores, key=lambda x: x[1], reverse=True)
        return [w for w, s in sorted_words[:top_n]]


