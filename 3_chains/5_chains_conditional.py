from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableBranch, RunnableLambda
import re

# Load environment variables from .env
load_dotenv()

# Create a ChatGroq model
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_retries=2
)

# Define prompt templates for different feedback types
positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human",
         "Generate a thank you note for this positive feedback: {feedback}."),
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human",
         "Generate a response addressing this negative feedback: {feedback}."),
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Generate a request for more details for this neutral feedback: {feedback}.",
        ),
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Generate a message to escalate this feedback to a human agent: {feedback}.",
        ),
    ]
)

# Define the feedback classification template
classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human",
         "Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}."),
    ]
)

# Create the classification chain
classification_chain = classification_template | model | StrOutputParser()

# Function to preserve the input
def classify_and_preserve_input(inputs):
    # Get classification
    classification = classification_chain.invoke(inputs)
    # Return both classification and original input
    return {"classification": classification, "feedback": inputs["feedback"]}

# Define the runnable branches for handling feedback
branches = RunnableBranch(
    (
        lambda x: "positive" in x["classification"],
        positive_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "negative" in x["classification"],
        negative_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "neutral" in x["classification"],
        neutral_feedback_template | model | StrOutputParser()
    ),
    escalate_feedback_template | model | StrOutputParser()
)

# Create the classification chain
classification_chain = classification_template | model | StrOutputParser()

# Combine classification and response generation into one chain
chain = RunnableLambda(classify_and_preserve_input) | branches

# Run the chain with an example review
# Good review - "The product is excellent. I really enjoyed using it and found it very helpful."
# Bad review - "The product is terrible. It broke after just one use and the quality is very poor."
# Neutral review - "The product is okay. It works as expected but nothing exceptional."
# Default - "I'm not sure about the product yet. Can you tell me more about its features and benefits?"

review = "The product is excellent. I really enjoyed using it and found it very helpful. I especially like the translation to Telugu feature"
result = chain.invoke({"feedback": review})

# Output the result
print(result)
