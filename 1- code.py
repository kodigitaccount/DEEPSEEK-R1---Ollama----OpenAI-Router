from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Define the prompt template
template = """Question: {question}

Answer: Let's think step by step."""

# Create the prompt template using the given template
prompt = ChatPromptTemplate.from_template(template)

# Initialize the model (use your desired model)
model = OllamaLLM(model="deepseek-r1")

# Create a chain with the prompt and the model
chain = prompt | model

# Capture user input
question = input("Enter your question here: ")

# If the user has entered a question, format the prompt and invoke the chain
if question:
    try:
        # Format the input question with the template
        formatted_prompt = prompt.format(question=question)
        
        # Pass the formatted prompt directly to the model (invoke chain)
        response = chain.invoke({"question": question})  # Pass it as a dictionary
        
        # Display the response from the model
        print("Response:", response)
    except Exception as e:
        print(f"Error: {e}")
