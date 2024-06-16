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
import operator
# import streamlit as st




# Define AgentState TypeDict 
class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    # st.write("response from AgentState is successful") 

# Define RAG function
async def perform_rag(query, llm, retriever_QA, settings):
    docs = retriever_QA.retriever.invoke({"question": query})
    context = "\n".join([doc.page_content for doc in docs])
    prompt_template = """Use the following context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Context: {context}
    Question: {question}
    Answer:"""
    try:
        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        chain = prompt | llm | StrOutputParser()
        # response = chain.invoke({"context": context, "question": query})
        response = await chain.ainvoke({"context": context, "question": query}, **settings)  # <-- Pass settings here!
        #response = await chain.ainvoke(context=context, question=query, **settings) 
        # st.write("response from perform_rag is successful") 
        return response
    except Exception as e:
        error_msg = f"Error in perform_rag: {e}"
        print(error_msg)  # Log the error for debugging
        return {"error": error_msg} # Return error information 

class RetrievalAgent:
    def __init__(self, llm, retriever_QA, settings):
        self.llm = llm
        self.retriever_QA = retriever_QA
        self.settings = settings # Store settings

    async def retrieve_relevant_info(self, input_data):
        try:
            input_text = input_data.get("input")
            if input_text is None:
                raise ValueError("Missing 'input' key in retrieve_relevant_info-input_data")
        # Retrieve relevant documents based on user query and generate response using RAG
        # response = perform_rag(query, self.llm, self.retriever_QA)  # Use perform_rag directly
        
        # Retrieve relevant documents based on user query and generate response using RAG
            response = await perform_rag(
                input_text, self.llm, self.retriever_QA, self.settings
            )  # Use perform_rag directly
        # st.write("response from RetrievalAgent is successful")
            return {"output": response}
        except Exception as e:
            error_msg = f"Error in RetrievalAgent: {e}"
            print(error_msg)  # Log the error for debugging
            return {"error": error_msg} # Return error information 

# Define a Fraud Detection Agent class
class FraudDetectionAgent:
    def __init__(self, llm, pandas_df_agent, settings):
        self.llm = llm
        self.agent = pandas_df_agent
        self.settings = settings # Store settings

    async def assess_fraud_risk(self, input_data):
        try:
            response = input_data.get("output")
            if response is None:
                raise ValueError("Missing 'response' key in assess_fraud_risk-response")

        # Analyze retrieved information and query to assess fraud risk
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
        # st.write("response from FraudDetectionAgent is successful") 
            risk_assessment = await chain.ainvoke({"response": response}, **self.settings)
            #risk_assessment = await chain.ainvoke(response=response, **self.settings)  
            # st.write("response from FraudDetectionAgent is successful")
            return {"output": risk_assessment}
        except Exception as e:
            error_msg = f"Error in FraudDetectionAgent: {e}"
            print(error_msg)  # Log the error for debugging
            return {"error": error_msg} # Return error information 

# Define an Explanation Agent class
class ExplanationAgent:
    def __init__(self, llm, settings):
        self.llm = llm
        self.settings = settings

    async def generate_explanation(self, input_data):
        try:
            risk_assessment = input_data.get("output")
            if risk_assessment is None:
                raise ValueError("Missing 'response' key in generate_explanation-risk_assessment")
        
            # Generate a user-friendly explanation of the fraud risk assessment
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
        #explanation = await chain.ainvoke(risk_assessment=risk_assessment, **self.settings)
        # st.write("response from ExplanationAgent is successful")
            return {"output": explanation}
        except Exception as e:
            error_msg = f"Error in ExplanationAgent: {e}"
            print(error_msg)  # Log the error for debugging
            return {"error": error_msg} # Return error information 
