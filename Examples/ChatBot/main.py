from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI()

# The input variable name that gets added in here is whatever we provide for that memory key option right there.
# So because we put in messages, the input set of input variables is gonna get this additional variable 
# called messages and whatever data our memory is storing that will be assigned to that key.
# return_message = True makes sure that in our set of input variables is not just a plain string. They are the human, system,
# AI messages that we have stored in our memory.
memory = ConversationBufferMemory(
    chat_memory=FileChatMessageHistory("messages.json"),
    memory_key="messages", 
    return_messages=True
)

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm=chat,
    prompt=prompt,
    memory=memory
) 

while True:
    content = input("You: ")

    result = chain({"content": content})

    print("ChatBot:", result.get("text", "Sorry, I am not sure how to respond to that."))
