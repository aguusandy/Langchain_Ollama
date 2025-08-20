from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing_extensions import Annotated, TypedDict
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph, END
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from typing import Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

class ChatBot:

    def __init__(self, thread_id: str, language: str = "English"):
        # model: llama3.2 by ollama
        self.model = init_chat_model("llama3.2", model_provider="ollama")
        # name of the thread 
        self.thread_id = thread_id
        # language of the conversation
        self.language = language
        # prompt template
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
            (
                "system",
                "You talk like a kind artificial assistant. You are a helpful assistant. You try to respond to every question with helpful information. Answer all questions to the best of your ability in {language}.",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        workflow = StateGraph(state_schema=State)

        workflow.add_edge(START, "model")
        workflow.add_node("model", self._call_model)
        # workflow.add_edge("model", END)
        
        self.memory = MemorySaver()
        self.app = workflow.compile(checkpointer=self.memory)


    def _call_model(self, state: State):
        prompt = self.prompt_template.invoke(state)
        response = self.model.invoke(prompt)
        return {"messages": [response]}
    
    # async def call_model(state: MessagesState):
        # response = await model.ainvoke(state["messages"])
        # return {"messages": response}

    def answer(self, question: str) -> str:
        # history = self.memory.get(self.thread_id)
        # if history is None or "messages" not in history:
        #     messages = []
        # else:
        #     messages = history["messages"]
        messages = []
        messages = messages + [HumanMessage(question)]
        print(f"Messages: {messages}")
        result = self.app.invoke(
            {"messages": messages, "language": self.language}, 
            {"configurable": {"thread_id": self.thread_id}}
        )
        # result = await self.app.invoke(
        #     {"messages": messages, "language": self.language}, 
        #     config={"configurable": {"thread_id": self.thread_id}}
        # )
        response = result["messages"][-1].pretty_print()
        return response
