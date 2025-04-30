import os
from langchain_chroma import Chroma
from langchain_groq import ChatGroq, GroqEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# Create a ChatGroq model
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_retries=2
)

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

# Load the stored vector database
embeddings = GroqEmbeddings()

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

# Define the user's question
query = "Where is Dracula's castle located?"

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.2},
)
relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    print(f"Source: {doc.metadata['source']}\n")


# combined_input = (
#     "Here are some documents that might help answer the question: "
#     + query
#     + "\n\nRelevant Documents:\n"
#     + "\n\n".join([doc.page_content for doc in relevant_docs])
#     + "\n\nPlease provide a rough answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
# )

# # Create a ChatOpenAI model
# model = ChatOpenAI(model="gpt-4o")

# # Define the messages for the model
# messages = [
#     SystemMessage(content="You are a helpful assistant."),
#     HumanMessage(content=combined_input),
# ]

# # print(messages, "messages")

# # Invoke the model with the combined input
# result = model.invoke(messages)

# # Display the full result and content only
# print("\n--- Generated Response ---")
# print("Full result:")
# # print(result)
# print("Content only:")
# print(result.content)
