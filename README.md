#  OCR + GenAI Text Insight Explorer

###  Extract, Analyze & Chat with Text from Images

Upload an image, analyze it using Various Features , and even **chat** with the extracted content using **Generative AI**.

---
#  Text Extraction and Analysis AI App

This project is an intelligent **Streamlit-based AI application** that enables users to upload any image (such as documents, articles, or handwritten notes) and automatically **extract meaningful text** using Optical Character Recognition (**OCR**) powered by *PyTesseract*.  

Once the text is extracted, the app performs a wide range of **Natural Language Processing (NLP)** operations — including **sentiment analysis**, **named entity recognition (NER)**, **keyword extraction**, **summarization**, and **tokenization** — all through cutting-edge models from **Hugging Face** and **spaCy**.  

In addition, it features a built-in **Generative AI Chat Interface**, allowing users to **ask questions or have an interactive conversation** with the extracted text. This is powered by **LangChain Agents** and **Hugging Face Transformers**, creating a seamless pipeline that connects OCR, NLP, and conversational AI into one unified, end-to-end intelligent text exploration platform.


##  Features

| Feature | Description |
|----------|-------------|
|  **OCR Extraction** | Extracts text from uploaded images using `PyTesseract`. |
|  **Text Cleaning** | Cleans unwanted symbols, `\n`, and non-ASCII characters. |
|  **Sentiment Analysis** | Detects emotional tone using a Hugging Face `sentiment-analysis` pipeline. |
|  **Named Entity Recognition (NER)** | Identifies names, locations, dates, etc. using **spaCy**. |
|  **Tokenization** | Splits text into words, digits, and punctuation. |
|  **Summarization** | Uses a Transformer-based summarizer to generate concise summaries. |
|  **Keyword Extraction** | Extracts key terms using `TfidfVectorizer`. |
|  **Question Answering** | Ask direct factual questions about the extracted text. |
|  **Chat with Extracted Text** | Have a conversation with your image’s content using **LangChain Agent** and **Hugging Face** models. |

---

##  Example Workflow

1. Upload an image (like a newspaper clipping, document, or article).  
2. Text is automatically extracted using **PyTesseract**.  
3. Explore multiple analysis tabs:
   - Sentiment, Tokens, Entities, Summary, Keywords
4. Ask context-based questions or **chat** with your extracted text.

---

##  Screenshots

###  OCR Extraction  
Displays extracted text from an uploaded image.

<img width="1822" height="869" alt="Screenshot 2025-11-05 060200" src="https://github.com/user-attachments/assets/5841bf9b-6030-445f-af8f-34c1bf263a20" />


---

###  Sentiment Analysis  
Analyzes emotional tone of the extracted text.

<img width="1804" height="712" alt="Screenshot 2025-11-05 060433" src="https://github.com/user-attachments/assets/796b2d37-72fb-46af-b814-20452366a3c5" />

---

###  Named Entity Recognition (NER)  
Highlights entities such as people, places, and organizations.

<img width="1787" height="840" alt="Screenshot 2025-11-05 060418" src="https://github.com/user-attachments/assets/6777f188-fb24-4336-8bc5-3cb581f13050" />


---

###  Text Summarization  
Summarizes lengthy text into concise and readable form.

<img width="1796" height="656" alt="Screenshot 2025-11-05 060445" src="https://github.com/user-attachments/assets/0d8c5b72-2bcd-431b-8f22-09512e1ebfaf" />


---

###  Keywords Extraction  
Finds the most relevant keywords using TF-IDF scores.

<img width="1800" height="775" alt="Screenshot 2025-11-05 063656" src="https://github.com/user-attachments/assets/2f76dc3a-86af-442d-a802-d3e33c1d603d" />


---

###  Chat with Extracted Text  
Chat dynamically with the extracted content using LangChain + Hugging Face agent.

<img width="1860" height="824" alt="Screenshot 2025-11-05 075606" src="https://github.com/user-attachments/assets/29789242-134c-4eb6-8ab6-095e2c5efd13" />


---

##  Frameworks & Libraries Used

###  **Core**
- [Streamlit](https://streamlit.io/) – Frontend for interactive UI  
- [Python 3.10+](https://www.python.org/)  
- [Pillow (PIL)](https://pillow.readthedocs.io/) – Image processing  
- [PyTesseract](https://pypi.org/project/pytesseract/) – OCR text extraction  

###  **NLP**
- [spaCy](https://spacy.io/) – Tokenization & Named Entity Recognition  
- [Transformers](https://huggingface.co/transformers/) – Sentiment, Summarization & QnA pipelines  
- [scikit-learn](https://scikit-learn.org/) – TF-IDF keyword extraction  

###  **Generative AI**
- [LangChain](https://www.langchain.com/) – For building conversational agents  
- [LangGraph](https://github.com/langchain-ai/langgraph) – Persistent memory in chat  
- [Hugging Face Hub](https://huggingface.co/) – Access to powerful language models  

---

##  Installation

```bash
# 1️⃣ Clone the repository
git clone https://github.com/<Harsh-Anand-20-06>/SnapTalk.git
cd SnapTalk

# 2️⃣ Create virtual environment
python -m venv .venv
source .venv/bin/activate  # (Linux/Mac)
.venv\Scripts\activate     # (Windows)

# 3️⃣ Install dependencies
pip install -r requirements.txt
