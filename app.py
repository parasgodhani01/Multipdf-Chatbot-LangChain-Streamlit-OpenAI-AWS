import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from config import process_pdf

# Load API key
load_dotenv(override=True)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") #type: ignore

def user_input(user_question):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = process_pdf.get_conversational_chain()
    context = "\n\n".join([d.page_content for d in docs])

    response = chain.invoke({"context": context, "question": user_question})
    st.write("### 🤖 Reply:")
    st.write(response.content)


def main():
    st.set_page_config("PDF Q&A Chatbot", page_icon="📘")
    st.header("📚 PDF Chatbot with OpenAI 🤖")

    user_question = st.text_input("Ask a question about your uploaded PDF:")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📂 Upload PDF files")
        pdf_docs = st.file_uploader("Upload PDF files here and click Submit", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Processing your PDFs..."):
                raw_text = process_pdf.get_pdf_text(pdf_docs)
                text_chunks = process_pdf.get_text_chunks(raw_text)
                process_pdf.get_vector_store(text_chunks)
                st.success("✅ Processing complete! You can now ask questions.")


if __name__ == "__main__":
    main()
