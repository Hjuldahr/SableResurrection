# New WIP testing interface
import atexit
import sys
import gradio as gr
from signal import SIGBREAK, SIGINT, SIGTERM, signal

from ai.intelligence_v3 import Sable

sable = None

def generate(message: str, attachments: list[str]) -> str:
    sable.submit(message, attachments)
    return sable.generate()

client = gr.Interface(
    generate,
    title="Sable",
    inputs=["textbox"],
    outputs=["text"],
    additional_inputs=[
        "files"
    ],
    api_name="predict",
)

if __name__ == '__main__':
    sable = Sable()
    
    atexit.register(sable.close)
    signal(SIGINT, sys.exit)
    signal(SIGTERM, sys.exit)
    signal(SIGBREAK, sys.exit)
        
    client.launch()