from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv
import argparse
import json

# Load environment variables from a .env file
load_dotenv()

# Set up argument parsing for command-line options
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers", help="The task to generate code for")
parser.add_argument("--language", default="python", help="The programming language for the code")
args = parser.parse_args()

# Initialize the OpenAI object from LangChain with API keys loaded from environment variables
llm = OpenAI()

# Create a PromptTemplate for generating code based on the task and language
code_prompt = PromptTemplate(
    input_variables=["task", "language"],
    template="Write a very short {language} function that will {task}."
)

# Create another PromptTemplate for generating tests for the code
test_prompt = PromptTemplate(
    input_variables=["language", "code"],
    template="Write a test for the following {language} code:\n{code}"
)

# Set up LLMChains for generating code and tests, with output keys specified to capture results
code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key="code"
)
test_chain = LLMChain(
    llm=llm,
    prompt=test_prompt,
    output_key="test"
)

# Create a SequentialChain to first generate code and then generate a test for that code
chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["task", "language"],
    output_variables=["code", "test"]
)

# Execute the SequentialChain with the provided task and language arguments
result = chain({
    "language": args.language,
    "task": args.task
})

# Print the generated code and test to the console
print(">>>>>> GENERATED CODE:  <<<<<<\n")
print(result["code"] or "No code generated.")

print(">>>>>> GENERATED TEST: <<<<<<\n")
print(result["test"] or "No test generated.")
