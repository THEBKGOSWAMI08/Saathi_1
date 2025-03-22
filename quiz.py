import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
import requests
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Missing Google API Key. Set GOOGLE_API_KEY in your .env file.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text
#hello
# Extract text from a URL
def fetch_url_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text()
    except requests.RequestException as e:
        st.error(f"Error fetching URL content: {e}")
        return ""

# Generate quiz questions with answers using AI
def generate_quiz_questions_from_text(text, noq):
    prompt_template = """
    Generate {NOQ} quiz questions based on the following content:
    {context}

    Provide questions with four answer options (A, B, C, D). Also, include the correct answer in this format: 
    **Answer:** [Correct Option]

    Format:
    Question 1: [Your question here]
    A. [Option A]
    B. [Option B]
    C. [Option C]
    D. [Option D]
    **Answer:** [Correct Answer]
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "NOQ"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    document = Document(page_content=text)
    response = chain({"input_documents": [document], "NOQ": noq})

    if response and "output_text" in response:
        return parse_generated_questions(response["output_text"])
    else:
        return []

# Parse AI-generated questions into a structured format
def parse_generated_questions(raw_text):
    questions = []
    current_question = {}

    lines = raw_text.split("\n")
    for line in lines:
        line = line.strip()

        if line.startswith("Question"):
            if current_question:
                questions.append(current_question)
            current_question = {"question": line, "options": [], "answer": None}

        elif line.startswith(("A.", "B.", "C.", "D.")):
            if current_question:
                current_question["options"].append(line)

        elif line.startswith("**Answer:**"):
            if current_question:
                current_question["answer"] = line.replace("**Answer:**", "").strip()

    if current_question:
        questions.append(current_question)

    return questions

# Run the Streamlit quiz app
def run_quiz():
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'> Quiz Generator 🎓</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px;'>Generate quiz questions from PDFs, Topics, or URLs </p>", unsafe_allow_html=True)

    option = st.selectbox("Select Input Type:", ("📄 Upload PDF", "💬 Enter Topic", "🔗 Enter URL"))
    noq = st.number_input("Enter Number of Questions:", min_value=1, max_value=50, value=5)

    # Initialize session state for user answers and quiz questions
    if "quiz_questions" not in st.session_state:
        st.session_state.quiz_questions = []
    if "user_answers" not in st.session_state:
        st.session_state.user_answers = {}

    # Fetch and generate questions
    if option == "📄 Upload PDF":
        pdf_docs = st.file_uploader("Upload your PDF Files:", accept_multiple_files=True)
        if st.button("Generate Quiz"):
            if pdf_docs:
                with st.spinner("Processing PDF..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text.strip():
                        st.session_state.quiz_questions = generate_quiz_questions_from_text(raw_text, noq)
                        st.session_state.user_answers = {}  # Reset user answers
                        st.success("Quiz Generated!")
                    else:
                        st.warning("No readable text found in the PDF.")
            else:
                st.warning("Please upload at least one PDF file.")
    
    elif option == "💬 Enter Topic":
        topic = st.text_area("Enter the Topic:")
        if st.button("Generate Quiz"):
            if topic.strip():
                with st.spinner("Generating Questions..."):
                    st.session_state.quiz_questions = generate_quiz_questions_from_text(topic, noq)
                    st.session_state.user_answers = {}  # Reset user answers
                    st.success("Quiz Generated!")
            else:
                st.warning("Please enter a valid topic.")
    
    elif option == "🔗 Enter URL":
        url = st.text_input("Enter the URL:")
        if st.button("Generate Quiz"):
            if url.strip():
                with st.spinner("Fetching URL Content..."):
                    content = fetch_url_content(url)
                    if content.strip():
                        st.session_state.quiz_questions = generate_quiz_questions_from_text(content, noq)
                        st.session_state.user_answers = {}  # Reset user answers
                        st.success("Quiz Generated!")
                    else:
                        st.warning("No readable content found at the URL.")
            else:
                st.warning("Please enter a valid URL.")

    # Display generated questions
    if st.session_state.quiz_questions:
        st.subheader("📝 Quiz Questions")
        for i, q in enumerate(st.session_state.quiz_questions):
            st.write(f"{q['question']}")
            
            # Store user-selected answers in session state
            st.session_state.user_answers[i] = st.radio(
                f"Select your answer for question {i+1}:",
                options=q["options"],
                index=None,  # No default selection
                key=f"question_{i}"
            )

        # Submit button
        if st.button("Submit Quiz"):
            if None in st.session_state.user_answers.values():
                st.warning("Please answer all questions before submitting.")
            else:
                score = 0
                incorrect_questions = []

                for i, q in enumerate(st.session_state.quiz_questions):
                    correct_answer = q["answer"]
                    user_answer = st.session_state.user_answers[i]

                    if user_answer == correct_answer:
                        score += 1
                    else:
                        incorrect_questions.append((q["question"], correct_answer, user_answer))

                # Display Score
                st.success(f"🎯 Your Score: {score}/{len(st.session_state.quiz_questions)}")

                # Show Mistakes & Areas to Improve
                if incorrect_questions:
                    st.subheader("🔍 Review Your Mistakes ❌")
                    for q_text, correct, user in incorrect_questions:
                        st.markdown(f"**Question:** {q_text}")
                        st.markdown(f"❌ **Your Answer:** {user}")
                        st.markdown(f"✅ **Correct Answer:** {correct}")
                        st.markdown("---")

                    st.subheader("📌 Areas to Improve")
                    topics = [q_text.split(":")[1].strip() if ":" in q_text else q_text for q_text, _, _ in incorrect_questions]
                    unique_topics = set(topics)
                    for topic in unique_topics:
                        st.write(f"🔹 Revise: {topic}")

if __name__ == "__main__":
    run_quiz()
