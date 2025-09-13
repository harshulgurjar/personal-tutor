import streamlit as st
import sqlite3
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
import tempfile
import torch

# Initialize conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ---------------- Database Functions ----------------
def init_db():
    conn = sqlite3.connect("students.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS student_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            level TEXT,
            goal TEXT,
            pace TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_student_profile(name, level, goal, pace):
    conn = sqlite3.connect("students.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM student_profiles WHERE name=?", (name,))
    row = cursor.fetchone()
    if row:
        cursor.execute("UPDATE student_profiles SET level=?, goal=?, pace=? WHERE name=?", (level, goal, pace, name))
    else:
        cursor.execute("INSERT INTO student_profiles (name, level, goal, pace) VALUES (?, ?, ?, ?)", (name, level, goal, pace))
    conn.commit()
    conn.close()

def get_student_profile(name):
    conn = sqlite3.connect("students.db")
    cursor = conn.cursor()
    cursor.execute("SELECT level, goal, pace FROM student_profiles WHERE name=?", (name,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"level": row[0], "goal": row[1], "pace": row[2]}
    return None

init_db()

# ---------------- Sidebar ----------------
st.sidebar.title("Student Profile")
name = st.sidebar.text_input("Name")
level = st.sidebar.selectbox("Level", ["Beginner", "Intermediate", "Advanced"])
goal = st.sidebar.text_input("Learning Goal", "Revise computer science basics")
pace = st.sidebar.selectbox("Preferred Pace", ["Slow", "Medium", "Fast"])

if st.sidebar.button("Save Profile"):
    save_student_profile(name, level, goal, pace)
    st.sidebar.success(f"Profile for {name} saved.")

if st.sidebar.button("Load Profile"):
    profile = get_student_profile(name)
    if profile:
        st.sidebar.success(f"Profile loaded: {profile}")
    else:
        st.sidebar.info("No profile found. Please set your details.")

st.sidebar.title("PDF Upload")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

st.sidebar.title("LLM Settings")
model_name = st.sidebar.text_input("HuggingFace Model", "gpt2")
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------- Load Model and Tokenizer ----------------
@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, torch_dtype=torch.float16)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

# ---------------- Main App ----------------
if uploaded_file and model_name:
    # Save PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # Extract text
    reader = PdfReader(pdf_path)
    text = " ".join([page.extract_text().replace("\n", " ") for page in reader.pages if page.extract_text()])

    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)

    # Load models
    tokenizer, model = load_model_and_tokenizer(model_name)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # LLM pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, temperature=0.7)
    llm = HuggingFacePipeline(pipeline=pipe)

    # Conversational Retrieval Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )

    # Streamlit Interface
    st.title("PDF Tutor with AI")
    st.header("Ask Questions about your PDF")
    question = st.text_input("Type your question here")

    if question:
        profile = get_student_profile(name)
        level_val = profile["level"] if profile else "Beginner"
        goal_val = profile["goal"] if profile else "General learning"
        pace_val = profile["pace"] if profile else "Medium"

        # Get answer
        result = qa_chain({"question": question})
        st.write("**Answer:**")
        st.write(result["answer"])

    # Quiz Generation
    st.header("Generate Multiple Choice Quiz")
    topic = st.text_input("Quiz Topic")
    if st.button("Create Quiz") and topic:
        quiz_template = """
You are a quiz generator AI tutor.
Create 3 multiple-choice questions on the topic below.
Topic: {topic}
"""
        quiz_prompt = PromptTemplate(input_variables=["topic"], template=quiz_template)
        quiz_chain = LLMChain(llm=llm, prompt=quiz_prompt)
        quiz = quiz_chain.run(topic=topic)
        st.write("**Quiz:**")
        st.markdown(quiz)

else:
    st.info("Upload a PDF and specify an LLM model to get started.")

st.sidebar.caption("Powered by LangChain, HuggingFace, FAISS, and Streamlit.")
