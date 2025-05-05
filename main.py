import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.messages import HumanMessage

load_dotenv()

groq_api_key = os.getenv('lang-graph-api')


llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="Llama3-8b-8192"  # "Gemma-2b-it", "Gemma-7b-it", "Mixtral-8x7b", "Llama3-8b-8192"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])


chain = prompt | llm

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: FileChatMessageHistory(f"history_{session_id}.json"),
    input_messages_key="input",
    history_messages_key="history"
)

session_id = "user1"

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    response = chain_with_history.invoke({"input": user_input}, config={"configurable": {"session_id": session_id}})
    print("AI:", response.content)


