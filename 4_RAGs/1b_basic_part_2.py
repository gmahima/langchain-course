import os
from langchain_chroma import Chroma
from langchain_groq import ChatGroq, GroqEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnablePassthrough

load_dotenv()

# Create a ChatGroq model
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_retries=2
)

# Load the previously created vector store
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Load the stored vector database
embeddings = GroqEmbeddings()

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

# Define the user's question
query = "Where does Gandalf meet Frodo?"

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 10, "score_threshold": 0.9}, 
)
relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

# Create the RAG prompt template
rag_prompt_template = ChatPromptTemplate.from_template(
    """Answer the following question based on the provided context:
    
    Context:
    {context}
    
    Question:
    {question}
    """
)

# Define the RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt_template
    | llm
    | StrOutputParser()
)

# Run the RAG chain
print("\n--- RAG Response ---")
response = rag_chain.invoke(query)
print(response)
