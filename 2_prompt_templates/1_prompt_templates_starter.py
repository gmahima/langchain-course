from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_retries=2
)

# template = "Write a {tone} email to {company} expressing interest in the {position} position, mentioning {skill} as a key strength. Keep it to 4 lines max"

# prompt_template = ChatPromptTemplate.from_template(template)

# prompt =  prompt_template.invoke({
#     "tone": "energetic", 
#     "company": "samsung", 
#     "position": "AI Engineer", 
#     "skill": "AI"
# })

# Example 2: Prompt with System and Human Messages (Using Tuples)
messages = [
    ("system", "You are a well versed reasearcher who tells facts about {topic}."),
    ("human", "Tell me {count} facts."),

]
template = "You are a well versed reasearcher in {topic}. Tell me {count} facts about {topic}."

prompt_template = ChatPromptTemplate.from_messages(messages)
second_prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({"topic": "lawyers", "count": 3})

second_prompt = second_prompt_template.invoke({"topic": "plants", "count": 1})
print(prompt)
# Output:
# messages=[
#     SystemMessage(content='You are a well versed reasearcher who tells facts about lawyers.', additional_kwargs={}, response_metadata={}),
#     HumanMessage(content='Tell me 3 facts.', additional_kwargs={}, response_metadata={})
# ]
print(second_prompt)
# Output:
# messages=[
#     HumanMessage(content='You are a well versed reasearcher in plants. Tell me 1 facts about plants.', additional_kwargs={}, response_metadata={})
# ]
# result = llm.invoke(prompt)

# print(result)  

