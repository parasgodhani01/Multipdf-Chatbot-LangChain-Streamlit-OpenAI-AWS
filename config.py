from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough


class process_pdf:

    @staticmethod
    def get_pdf_text(pdf_docs):
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        return text

    @staticmethod
    def get_text_chunks(text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        return chunks

    @staticmethod
    def get_vector_store(text_chunks):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")

    @staticmethod
    def get_conversational_chain():
        # Define the prompt
        prompt_template = """
        You are a helpful assistant. Use the provided context to answer the question.
        If the answer is not in the context, respond with "Answer not available in the provided context."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """

        # LLM model
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

        # Create prompt object
        prompt = PromptTemplate.from_template(prompt_template)

        # Build the new Runnable pipeline
        chain = RunnableMap({
            "context": RunnablePassthrough(),
            "question": RunnablePassthrough()
        }) | prompt | llm

        return chain
