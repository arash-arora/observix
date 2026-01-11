from langchain_core.messages import HumanMessage, SystemMessage

from observix.llm.langchain import ChatGroq

message = ChatGroq(model="openai/gpt-oss-120b").invoke([
    SystemMessage(content="Hello, world!"),
    HumanMessage(content="Hello, world!")
])
print(message)  