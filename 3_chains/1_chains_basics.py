from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# Load environment variables from .env
load_dotenv()

# Create a ChatGroq model
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_retries=2
)

# Define prompt templates (no need for separate Runnable chains)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a facts expert who knows facts about {animal}."),
        ("human", "Tell me {fact_count} facts."),
    ]
)

# Create the combined chain using LangChain Expression Language (LCEL)
chain = prompt_template | model | StrOutputParser()
# chain = prompt_template | model

# Run the chain
result = chain.invoke({"animal": "elephant", "fact_count": 1})

# Output
print(result)
