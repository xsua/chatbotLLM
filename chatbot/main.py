import gra
from fastapi import FastAPI,WebSocket,WebSocketDisconnect,Request
from gradio_client.documentation import document
from langchain_community.llms import Tongyi
from langchain.agents import create_openai_tools_agent,AgentExecutor,tool
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.utilities import GoogleSerperAPIWrapper
from numpy.array_api import result_type
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.memory import ConversationTokenBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
import uuid
app = FastAPI()

sessions={}

@tool
def search(query:str):
    """需要实时搜索的时候才会使用此工具"""
    serp=GoogleSerperAPIWrapper(serper_api_key="")
    result=serp.run(query)
    return  result

@tool
def get_loc_info(qurey:str):
    """回答问题的时候，优先使用此工具"""
    Qdrant_client=Qdrant(
        QdrantClient(path="/loc_qdrand"),
        "local_documents",
        DashScopeEmbeddings(),

    )
    retriever=Qdrant_client.as_retriever(search_type="simiarity")
    result=retriever.get_relevant_documents(query)
    return result



class Chatbot:
    def __init__(self):
        self.chatmodel=Tongyi(
            model_name="qwen-2.5-3B-Instruct",
            temperature=0,
            streaming=True,
            api_key="sk-9355a718d0f547fb88fb9d930f2bf33b"
        )
        self.MEMEORY_KEY="chat_history"
        self.SYSTEMPL='''你是一名十分专业的中小学高级老师，别人都叫你小松老师。
        以下是你的个人角色设定：
        1、你擅长回答孩子教育、中小学书籍推荐、松鼠AI学习机导购等方面的问题。
        2、你今年35岁，性别女，家住在中国山东省，长相美丽动人，性格开朗大方。
        3、你总是用简体中文来作答。
        4、你从不说自己是一个人工智能助手或AI。
        5、语言热情友好、通俗易懂，避免使用过于专业或晦涩难懂的术语，确保顾客能够轻松理解。
        问题理解与分析策略：
        1、当顾客提出问题时，迅速提取关键词，判断是关于产品功能咨询、型号比较、价格询问、适用人群匹配还是购买流程相关。
        2、对于复杂的表述，仔细剖析句子结构与逻辑，确定顾客的核心需求与潜在关注点，如顾客说 “我孩子上初中，数学不好，
        想要个学习机能帮忙提高成绩，价格别太贵”，要明确顾客的孩子所处学习阶段、学科短板以及预算限制等要点，以便给出精准回应。
        回答框架：
        1、针对功能咨询类问题，先简要介绍相关功能的基本原理与作用，再结合实际学习场景举例说明其优势与效果。
        2、对于型号比较问题，采用表格或分点对比的形式呈现不同型号在关键方面（如硬件、功能、价格）的差异。
        3、涉及价格询问时，清晰准确地告知顾客各型号的价格范围以及当前是否有优惠活动、优惠后的实际价格等信息。
        4、若遇到不理解的问题或超出自身知识范围的问题，你会使用搜索工具来搜索。
        5、当顾客提出模糊不清的问题时，通过追问来明确问题意图。
        6、你会保存每一次的聊天记录，以便在后续的对话使用。
        7、你只使用简体中文来作答，否则你会受到惩罚。
        '''
        self.prompt=ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    self.SYSTEMPL,
                ),
                MessagesPlaceholder(variable_name=self.MEMEORY_KEY),
                (
                    "user",
                    "{input}"
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ],
        )
        tools=[search,get_loc_info]
        agent=create_openai_tools_agent(
            self.chatmodel,
            tools=tools,
            prompt=self.prompt,
        )

        self.memory =self.get_memory()

        memory=ConversationTokenBufferMemory(
            llm=self.chatmodel,
            human_prefix="用户",
            ai_prefix="小松老师",
            memory_key=self.MEMEORY_KEY,
            output_key="output",
            return_messages=True,
            max_token_limit=1000,
            chat_memory=self.memory,

        )
        self.agent_executor=AgentExecutor(
            agent=agent,
            Memory=memory,
            tools=tools,
        )


    def get_memory(self):
        chat_message_history=RedisChatMessageHistory(
        url="redis://localhost:6379/0",session_id=str(uuid.uuid4())
        )
        self.search=GoogleSerperAPIWrapper()

    def run(self,query):
        result=self.agent_executor.invoke({"input":query})
        return result

@app.post("/chat")
def chat(query:str):
    chatbot=Chatbot()
    return chatbot.run(query)

@app.post("/add_urls")
def add_urls(URL:str):
    loader=WebBaseLoader(URL)
    docs=loader.load()
    document=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
    ).split_documents(docs)
    qdrant=Qdrant.from_documents(
        document,
        DashScopeEmbeddings(model="", dashscope_api_key=""),
        path="",
        collection_name="",
    )
    return {"response":"urls added！"}

@app.post("/add_pdfs")
def add_pdfs():
    return {"response":"pdfs added！"}

@app.websocket("/ws")
async def websocket_endpoint(websocket:WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    sessions[session_id] = websocket
    try:
        while true:
            data=await websocket.receive_text()
            await websocket.send_text(f"你发送的信息：{data}")
    except WebSocketDisconnect:
        print("发生错误")
    finally:
        if session_id in sessions:
            del sessions[session_id]
            await websocket.close()


if __name__=="__main__":

    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port=8000)


