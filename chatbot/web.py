from fastapi import FastAPI
import gradio as gr

import main

#CUSTOM_PATH = "/gradio"

app = FastAPI()

@app.get("/")
def read_main():
    return {"message": "This is your main app"}

def create_gradio():
    with gr.Blocks() as web:
        gr.Markdown("欢迎使用")
        input_text=gr.Textbox(label="输入")
        button=gr.Button("提交")
        output_text=gr.Textbox(label="输出")
        button.click(fn=main.chat,inputs=input_text,outputs=output_text)

    return web

if __name__=="__main__":
    import uvicorn
    app = gr.mount_gradio_app(app,create_gradio(), path="/chat")
    uvicorn.run(app, host="127.0.0.1", port=8012)

