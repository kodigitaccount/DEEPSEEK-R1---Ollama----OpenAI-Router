from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import streamlit as st

# Set the app title
st.title("NareshIT bot using DeepSeek-R1")

# Define the prompt template
template = """Question: {question}

Answer: Let's think step by step."""

# Create the prompt template using the given template
prompt = ChatPromptTemplate.from_template(template)

# Initialize the model (uncomment the correct model depending on your choice)
# model = OllamaLLM(model="llama3.1")
model = OllamaLLM(model="deepseek-r1")

# Create a chain with the prompt and the model
chain = prompt | model

# Use text_input to capture user question
question = st.text_input("Enter your question here")

# If the user has entered a question, format the prompt and invoke the chain
if question:
    try:
        # Format the input question with the template
        formatted_prompt = prompt.format(question=question)
        
        # Pass the formatted prompt directly to the model
        response = chain.invoke(formatted_prompt)
        
        # Display the response from the model
        st.write(response)
    except Exception as e:
        st.write(f"Error: {e}")
