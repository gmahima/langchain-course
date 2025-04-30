from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda, RunnableSequence

# Load environment variables from .env
load_dotenv()

# Create a ChatGroq model
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_retries=2
)

# Define prompt templates
prompt_template = ChatPromptTemplate.from_messages(
     [
        ("system", "You love facts and you tell facts about {animal}"),
        ("human", "Tell me {count} facts."),
    ]
)

# Create individual runnables (steps in the chain)
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x)) # returns a PromptValue object - has the vars replaced with the actual values
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages())) # returns a normal list of ChatMessage objects
parse_output = RunnableLambda(lambda x: x.content) # returns a string

# Create the RunnableSequence (equivalent to the LCEL chain)
chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

# Run the chain
response = chain.invoke({"animal": "cat", "count": 1})

# Output
print(response)

