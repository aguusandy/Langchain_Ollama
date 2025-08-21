import os
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RagBot:
    def __init__(self, model_name="llama3.2", embed_model="mxbai-embed-large", db_location="./chrome_langchain_db"):
        # LLM and prompt
        self.model = OllamaLLM(model=model_name)
        self.template = """
            You are an assistant that answers questions based on the provided context. The context is about files and documents.
            You should answer based in the context and you don't try to invent false information. If you can't answer the question, you retrive a response saying that you don't have the information to answer that.

            Context:
            {context}

            Question: {question}

            Answer:
        """
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.chain = self.prompt | self.model
        self.db_location = db_location
        self.embeddings = OllamaEmbeddings(model=embed_model)
        self.vector_store = None
        self.retriever = None
        self.documents = []
        self.texts = []

    def load_pdfs(self, pdf_files):
        add_documents = not os.path.exists(self.db_location)
        print(f"add_documents: {add_documents}")
        if add_documents:
            documents = []
            ids = []
            for i, pdf_file in enumerate(pdf_files):
                if os.path.exists(pdf_file):
                    loader = PyPDFLoader(pdf_file)
                    loaded = loader.load()
                    documents.extend(loaded)
                    ids.append(i)
                    print(f"Loaded {pdf_file}: {len(loaded)} pages")
                else:
                    print(f"File not found: {pdf_file}")
            print(f"Total pages of documents loaded: {len(documents)}")
            self.documents = documents
            self.split_loaded_documents()
        else:
            print("Documents already loaded")
            self.documents = []
        self.store_chunks()
        return self.documents

    def split_loaded_documents(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.texts = text_splitter.split_documents(self.documents)
        # print(f"Total text chunks created: {len(self.texts)}")
        return self.texts

    def store_chunks(self):
        self.vector_store = Chroma(
            collection_name="pdf_seeker",
            persist_directory=self.db_location,
            embedding_function=self.embeddings
        )
        if self.documents:
            self.vector_store.add_documents(documents=self.documents)
        self.retriever = self.vector_store.as_retriever()

    def ask(self, question):
        if not self.retriever:
            raise Exception("Retriever not initialized. Call store_chunks() first.")
        search = self.retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in search])
        result = self.chain.invoke({"context": context, "question": question})
        return result

    # def interactive(self):
    #     while True:
    #         question = input("Enter your question (q to quit): ")
    #         if question.lower() == "q":
    #             break
    #         print(self.ask(question))

