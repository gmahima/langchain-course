from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv() 
#loads .env file

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_retries=2
)

result = llm.invoke("HI there. Who are you?")

print(result)


print("Hello World")

