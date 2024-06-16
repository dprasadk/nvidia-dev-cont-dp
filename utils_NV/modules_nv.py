# Importing essential libraries
import os
import getpass
 

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema.output_parser import StrOutputParser

from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from typing import Optional  # Import Optional from typing
import operator
# import streamlit as st




# Define AgentState TypeDict 
class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    error: Optional[str]  # Add an 'error' field to AgentState

# Define RAG function
#async def perform_rag(query, llm, retriever_QA, settings):
async def perform_rag(retriever_QA, llm, query, settings):
    print("Inside perform_rag - Query:", query)
    #print("retriever_QA type:", type(retriever_QA))
    #print("retriever_QA.retriever type:", type(retriever_QA.retriever))

    try: 

        try:
            # --- Retrieval --- 
            #docs =  retriever_QA.retriever.invoke({"question": query})
            #docs =  retriever_QA.retriever.invoke({"question": query})
            # Direct retrieval using retriever 
            docs = retriever_QA.invoke(query)  
            print("Retrieved Documents:", docs) 
            #context = "\n".join([doc.page_content for doc in docs])

             # --- Construct context from retrieved Documents --- 
            context = ""
            for i, doc in enumerate(docs):
                print(f"--- Document {i + 1} ---")
                print("Type of doc:", type(doc))
                print("Document Content:", doc.page_content)
                print("Document Metadata:", doc.metadata)
                context += doc.page_content + "\n"  # Add each document to context
            
        except Exception as e:
            error_msg = f"Error during perform_rag retrieval: {e}"
            print(error_msg) 
            return {"error": error_msg} 


        # --- Prompt Construction ---
        prompt_template = """Use the following context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Context: {context}
        Question: {question}
        Answer:"""
   
        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
            )
        
        
         # --- Direct LLM Call (for isolation testing - Uncomment to test directly) --- 
        """
        headers = {
            'Authorization': f'Bearer {os.environ["NVIDIA_API_KEY"]}',
            'Content-Type': 'application/json'
        }
        data = { 
            "prompt": prompt,  
            "max_tokens": 512
        } 
        response = requests.post(
            f"https://integrate.api.nvidia.com/v1/inference/mistralai/mixtral-8x22b-instruct-v0.1",  # Verify URL!
            headers=headers, 
            json=dataCheck 
        ) 

        if response.status_code == 200: 
            print("Direct LLM Call Successful:", response.json()) 
            return response.json() # Process the successful response
        else:
            print("Direct LLM Call Error:", response.status_code, response.text)
            raise Exception(f"Direct LLM Call Failed: {response.text}") 
 
        """      
        # --- LLM Invocation (with Enhanced Logging) ---
        try:
            chain = prompt | llm | StrOutputParser()
            response = await chain.ainvoke({"context": context, "question": query}, **settings)  # <-- Pass settings here!
            print("LLM Response:", response)
            
        except Exception as e:
            print("Error during LLM inference:", e)  # Log LLM-specific errors
            raise  # Re-raise the exception to be caught by the outer try...except

        return response
        
    except Exception as e:
        error_msg = f"Error in perform_rag: {e}"
        print(error_msg)  # Log the error for debugging
        return {"error": error_msg} # Return error information 

# Define an Retrieve Agent class
class RetrievalAgent:
    def __init__(self, llm, retriever_QA, settings):
        self.llm = llm
        self.retriever_QA = retriever_QA
        self.settings = settings # Store settings

    async def retrieve_relevant_info(self, input_data: AgentState) -> AgentState:
        try:
            input_text = input_data.get("input")
            if input_text is None:
                raise ValueError("Missing 'input' key in retrieve_relevant_info-input_data")

            # --- Debugging: Input Inspection --- 
            print("Input to perform_rag (Type):", type(input_text))
            print("Input to perform_rag (Content):", input_text)

            # --- Debugging: RAG Pipeline Check ---
            print("Entering perform_rag function...")  # Add a marker
            #response = await perform_rag(
             #   input_text, self.llm, self.retriever_QA, self.settings
            #)
            response = await perform_rag(
                self.retriever_QA, 
                self.llm, 
                input_text, 
                self.settings
                )  # Pass llm and settings
            # Check for errors from perform_rag
            if "error" in response:
                raise ValueError(response["error"]) 

            print("Exiting perform_rag function...") # Add a marker to see if it's reached
            print("Response from perform_rag:", response)

            # --- Ensure 'response' is stored as 'output' ---
            input_data["agent_outcome"] = AgentFinish(
                return_values={"output": response},  
                log="Retrieved relevant information."
            )
            return input_data 

        except Exception as e:
            error_msg = f"Error in RetrievalAgent: {e}"
            print(error_msg)
            input_data["error"] = error_msg  # Store error in AgentState
            input_data["agent_outcome"] = AgentFinish(
                return_values={"error": error_msg},
                log="Error during retrieval."
            )
            return input_data


# Define a Fraud Detection Agent class

class FraudDetectionAgent:
    def __init__(self, llm, pandas_df_agent, settings):
        self.llm = llm
        self.agent = pandas_df_agent  # Consider if this is actually used
        self.settings = settings 

    async def assess_fraud_risk(self, input_data: AgentState) -> AgentState:
        
        if input_data.get("error"):  # Check for prior errors
            print("FraudDetectionAgent - Skipping due to prior error:", input_data["error"])
            return input_data

        try:
            # --- Debugging: Input Inspection ---
            print("FraudDetectionAgent - Input Data:", input_data)
            print("FraudDetectionAgent - Input Data Type:", type(input_data))

            response = input_data["agent_outcome"].return_values.get("output")

            # --- Debugging: Check if 'response' is present ---
            if response is None:
                print("FraudDetectionAgent - 'response' key is missing!")
                raise ValueError("Missing 'response' key in assess_fraud_risk-response")

            print("FraudDetectionAgent - Response:", response)

            # --- Analyze retrieved information and query to assess fraud risk ---
            prompt_template = """
            Given the following user query and retrieved information about check fraud patterns, assess the risk of fraud:

            Response: {response}

            Respond with:
            - High, Medium, or Low risk assessment
            - Explanation detailing the reasoning and red flags (if any)
            """

            prompt = PromptTemplate(
                input_variables=["response"], template=prompt_template
            )
            chain = prompt | self.llm | StrOutputParser()

            # --- Execute the Chain ---
            risk_assessment = await chain.ainvoke({"response": response}, **self.settings)

            # --- Debugging: Output Check ---
            print("FraudDetectionAgent - Risk Assessment:", risk_assessment)

            # --- Update AgentState with the risk assessment ---
            input_data["agent_outcome"] = AgentFinish(
                return_values={"output": risk_assessment},
                log="Assessed fraud risk."
            )
            return input_data

        except Exception as e:
            error_msg = f"Error in FraudDetectionAgent: {e}"
            print(error_msg)  
            input_data["error"] = error_msg  # Store error in AgentState
            # Return error information within AgentState
            input_data["agent_outcome"] = AgentFinish(
                return_values={"error": error_msg},
                log="Error during fraud risk assessment."
            )
            return input_data

# Define an Explanation Agent class

class ExplanationAgent:
    def __init__(self, llm, settings):
        self.llm = llm
        self.settings = settings

    async def generate_explanation(self, input_data: AgentState) -> AgentState:

        if input_data.get("error"): # Check for prior errors
            print("ExplanationAgent - Skipping due to prior error:", input_data["error"])
            return input_data 

        try:
            # --- Debugging: Input Inspection ---
            print("ExplanationAgent - Input Data:", input_data)
            print("ExplanationAgent - Input Data Type:", type(input_data))

            risk_assessment = input_data["agent_outcome"].return_values.get("output")

            # --- Debugging: Check if 'risk_assessment' is present ---
            if risk_assessment is None:
                print("ExplanationAgent - 'risk_assessment' key is missing!")
                raise ValueError(
                    "Missing 'risk_assessment' key in generate_explanation-input_data"
                )

            print("ExplanationAgent - Risk Assessment:", risk_assessment)

            # --- Generate a user-friendly explanation ---
            prompt_template = """
            Explain the following check fraud risk assessment to a bank customer:

            Risk Assessment: {risk_assessment}

            Include the following in your explanation:
            - Clarity and conciseness
            - Specific red flags mentioned in the assessment
            - Additional information on fraud prevention if relevant
            """
            prompt = PromptTemplate(
                input_variables=["risk_assessment"], template=prompt_template
            )
            chain = prompt | self.llm | StrOutputParser()

            explanation = await chain.ainvoke(
                {"risk_assessment": risk_assessment}, **self.settings
            )

            # --- Debugging: Output Check ---
            print("ExplanationAgent - Explanation:", explanation)

            # --- Update AgentState with the explanation ---
            input_data["agent_outcome"] = AgentFinish(
                return_values={"output": explanation},
                log="Generated explanation."
            )
            return input_data

        except Exception as e:
            error_msg = f"Error in ExplanationAgent: {e}"
            print(error_msg)  # Log the error for debugging
            input_data["error"] = error_msg # Store error in AgentState
            # Return error information within AgentState
            input_data["agent_outcome"] = AgentFinish(
                return_values={"error": error_msg},
                log="Error during explanation generation."
            )
            return input_data