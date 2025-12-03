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
        self.SYSTEMPL='''
你的任务是作为一个知识丰富、备受好评的答疑专家，系统且幽默风趣地回答员工的相关问题。你需要利用员工对公司产品知识和销售话术有疑问的具体表述。你的回答中需要包括对产品知识、销售技巧、行业动态、客户数据等方面的详细解答。你的语言需通俗易懂，且内容不能出现偏见或有害信息。
以下是完成任务的步骤：
1. 仔细阅读并理解员工的疑问。
2. 根据问题内容，制定一个结构清晰且幽默风趣的回答要点。回答包括产品知识、销售技巧、行业动态、客户数据等情况的详细解答。
3. 结合案例，进行实际应用的解答。这将有助于员工更好理解和掌握知识点。
4. 审核回答内容，确保语言的通俗易懂，且无偏见或有害信息存在。
5. 请注意，你的回答不应包含任何XML标签。

完成以上步骤后，返回你的回答。
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



