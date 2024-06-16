import os
import asyncio
import chainlit as cl
from openai import OpenAI as AsyncOpenAI
from dotenv import load_dotenv
from IPython.display import Image

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langgraph.graph import StateGraph, END



# Import your custom agent classes
from utils_NV.modules_nv import (
    RetrievalAgent,
    FraudDetectionAgent,
    ExplanationAgent,
    AgentState,
)
from utils_NV import config_nv

import logging
import getpass
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Configure logging (optional but recommended)
logging.basicConfig(level=logging.INFO)

# Securely access the NVIDIA API key
nvidia_api_key = os.getenv("NVIDIA_API_KEY")
if not nvidia_api_key:
    raise ValueError("NVIDIA_API_KEY environment variable is not set.")

# Initialize the OpenAI client with NVIDIA endpoint
client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1", 
    api_key=nvidia_api_key 
)

# Configure LLM settings
settings = {
    "model": "meta/llama3-8b-instruct", 
    "temperature": 0.5,
    "top_p": 1,
    "max_tokens": 1024,
}

# --- LangChain & LangGraph setup (modified for clarity) --- 

# Load LLM and embedding models from NVIDIA
llm = ChatNVIDIA(model="meta/llama3-8b-instruct", nvidia_api_key=os.environ["NVIDIA_API_KEY"], max_tokens=1024)
#embedder = NVIDIAEmbeddings(model="NV-Embed-QA")
embedder = NVIDIAEmbeddings(model="NV-Embed-QA")

"""
# Test Embedding 
text = "Hello"
try: 
    embeddings = embedder.embed_query(text)
    print("Embeddings:", embeddings)
except Exception as e:
    print(f"Error during embedding: {e}")
"""

# Load CSV Data and Create Pandas DataFrame Agent
try:
    loader = CSVLoader(file_path=config_nv.CHECK_FRAUD_PATTERNS_CSV)
    documents = loader.load()
    df = pd.read_csv(config_nv.CHECK_FRAUD_PATTERNS_CSV)
    pandas_df_agent = create_pandas_dataframe_agent(llm, df, verbose=True)
except FileNotFoundError as e:
    logging.error(f"Error loading CSV: {e}")

# Create FAISS index for retrieval
text_splitter = CharacterTextSplitter(chunk_size=500, separator="\n")
docs = text_splitter.split_documents(documents)
try:
    faiss_index = FAISS.from_documents(docs, embedder)
    faiss_index.save_local(config_nv.FAISS_INDEX_PATH)
except Exception as e: 
    logging.error(f"Error creating FAISS index: {e}")

try:
    faiss_index = FAISS.load_local(config_nv.FAISS_INDEX_PATH, embedder, allow_dangerous_deserialization=True)
    #faiss_index = FAISS.load_local(config_nv.FAISS_INDEX_PATH, embedder)
except FileNotFoundError as e:
    logging.error(f"Error loading FAISS index: {e}")

'''
# Test Query
test_query = "hi" 
try:
    docs_and_scores = faiss_index.similarity_search(test_query, k=3)  # Fetch top 3
    print("Type of docs_and_scores:", type(docs_and_scores))  # Check the type
    
    for i, doc in enumerate(docs_and_scores): # Iterate directly through documents
        print(f"--- Result {i + 1} ---")
        print("Type of doc:", type(doc))        
        print("Document Content:", doc.page_content) 
        print("Document Metadata:", doc.metadata)   # Access metadata like this 

        # --- Accessing the relevance score (if it's stored in metadata) ---
        score = doc.metadata.get("score")  
        if score:
            print("Score:", score)
            
except Exception as e:
    print(f"Error during FAISS retrieval: {e}")
'''
"""
# Create Retriever 
try:
    retriever_QA = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=faiss_index.as_retriever()
    )
    print("retriever_QA:", retriever_QA)
except Exception as e:
    error_msg = f"Error in app.py retriever_QA: {e}"
    print(error_msg)  # Log the error for debugging
"""
# Get the retriever directly
retriever_QA=faiss_index.as_retriever()
# Initialize agents 
retrieval_agent = RetrievalAgent(llm, retriever_QA, settings)
fraud_detection_agent = FraudDetectionAgent(llm, pandas_df_agent, settings)
explanation_agent = ExplanationAgent(llm, settings)

# Define LangGraph application
graph_vc = StateGraph(AgentState)
graph_vc.add_node("orchestrator", retrieval_agent.retrieve_relevant_info) 
graph_vc.add_node("fraud_detection", fraud_detection_agent.assess_fraud_risk)
graph_vc.add_node("explanation", explanation_agent.generate_explanation)
graph_vc.set_entry_point("orchestrator")
graph_vc.add_edge("orchestrator", "fraud_detection")
graph_vc.add_edge("fraud_detection", "explanation")
graph_vc.add_edge("explanation", END) 
app = graph_vc.compile()

# Save the image
image_data = app.get_graph().draw_png()
with open("my_graph.png", "wb") as f:
    f.write(image_data)

# --- Chainlit Integration ---

cl.instrument_openai()

@cl.on_chat_start
def start_chat():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant."}],
    )

    cl.Message(
        content="# Multi-Agent AI for Check Fraud Detection\n\nWelcome to the AI-powered check fraud detection assistant.",
        ).send()

async def async_generator(sync_gen):
    for item in sync_gen:
        yield item

@cl.on_message
async def main(message: cl.Message):
    message_history = cl.user_session.get("message_history", [])

     # ---  Extract User Input  ---
    user_input = message.content 
    message_history.append({"role": "user", "content": user_input}) 
    print(message_history)
    # message_history.append({"role": "user", "content": message.content})

    try: 
        print(user_input)
        #response = app.invoke({"input": message.content})
        response = await app.ainvoke({"input": user_input})

        # --- Handle Errors from Retrieval More Explicitly ---
        if response.get("error"):
            await cl.Message(content=f"An error occurred during retrieval: {response['error']}").send() 
            return  # Stop processing if there's a retrieval error
    
    # Access the final output directly (Corrected)
        output_text = response["agent_outcome"].return_values.get("output", "No response generated.") 

        msg = cl.Message(content="") 
        await msg.send()

        for chunk in output_text.split(): 
    
            await msg.stream_token(chunk + " ")
            await asyncio.sleep(0.02) 

        message_history.append({"role": "assistant", "content": output_text})
        await msg.update()

    except Exception as e: 
        await cl.Message(content=f"An error occurred: {e}").send()  