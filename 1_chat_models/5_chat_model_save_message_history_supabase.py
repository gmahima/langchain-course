# Example using Supabase for chat history storage with Groq LLM

from dotenv import load_dotenv
import os
from supabase import create_client
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

"""
Steps to replicate this example:
1. Create a Supabase account and project: https://supabase.com/
2. Set up the database schema with the SQL provided in the README or comments below
3. Get your Supabase URL and API key
4. Create a .env file with SUPABASE_URL, SUPABASE_KEY, and GROQ_API_KEY
5. pip install supabase langchain-groq

SQL to execute in Supabase SQL Editor:
```
-- Enable Row Level Security
create extension if not exists "uuid-ossp";

-- Create a table for chat messages
create table chat_messages (
  id uuid primary key default uuid_generate_v4(),
  session_id text not null,
  user_id text not null default 'secure_sample_user_123', -- Hardcoded user ID for sample project
  role text not null, -- 'user' or 'assistant'
  content text not null,
  created_at timestamp with time zone default now()
);

-- Enable Row Level Security
alter table chat_messages enable row level security;

-- Create a policy that only allows reading/writing messages that belong to the specified user_id
create policy "User can CRUD their own messages" 
on chat_messages
using (user_id = 'secure_sample_user_123');

-- Create a function to add a user message
create or replace function add_user_message(
  p_session_id text,
  p_content text
) returns uuid language sql security definer as $$
  insert into chat_messages (session_id, role, content)
  values (p_session_id, 'user', p_content)
  returning id;
$$;

-- Create a function to add an AI message
create or replace function add_ai_message(
  p_session_id text,
  p_content text
) returns uuid language sql security definer as $$
  insert into chat_messages (session_id, role, content)
  values (p_session_id, 'assistant', p_content)
  returning id;
$$;

-- Create a function to get messages for a session
create or replace function get_session_messages(
  p_session_id text
) returns table (
  id uuid,
  role text,
  content text,
  created_at timestamp with time zone
) language sql security definer as $$
  select id, role, content, created_at
  from chat_messages
  where session_id = p_session_id
  order by created_at asc;
$$;
```
"""

load_dotenv()

# Setup Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SESSION_ID = "user_session_new"  # This could be a username or a unique ID

# Initialize Supabase Client
print("Initializing Supabase Client...")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Functions to interact with chat history
def add_user_message(session_id, content):
    return supabase.rpc('add_user_message', {
        'p_session_id': session_id, 
        'p_content': content
    }).execute()

def add_ai_message(session_id, content):
    return supabase.rpc('add_ai_message', {
        'p_session_id': session_id, 
        'p_content': content
    }).execute()

def get_messages(session_id):
    response = supabase.rpc('get_session_messages', {
        'p_session_id': session_id
    }).execute()
    return response.data

# Get current chat history
print("Retrieving Chat History...")
chat_messages = get_messages(SESSION_ID)
print("Chat History Initialized.")

# Convert to LangChain message format
messages = []
for msg in chat_messages:
    if msg['role'] == 'user':
        messages.append(HumanMessage(content=msg['content']))
    elif msg['role'] == 'assistant':
        messages.append(AIMessage(content=msg['content']))

print("Current Chat History:", messages)

# Initialize Chat Model (Groq instead of OpenAI)
model = ChatGroq(model="llama-3.1-8b-instant")

print("Start chatting with the AI. Type 'exit' to quit.")

while True:
    human_input = input("User: ")
    if human_input.lower() == "exit":
        break

    # Add user message to Supabase
    add_user_message(SESSION_ID, human_input)
    
    # Add to current message list
    messages.append(HumanMessage(content=human_input))
    
    # Get AI response
    ai_response = model.invoke(messages)
    
    # Add AI message to Supabase
    add_ai_message(SESSION_ID, ai_response.content)
    
    # Add to current message list
    messages.append(ai_response)

    print(f"AI: {ai_response.content}")
