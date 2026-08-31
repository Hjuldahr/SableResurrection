from llama_cpp import Llama

class ToolManager:
    def __init__(self, path: str):
        self.llm = None
        self.path = path
    
    def __enter__(self):
        self.llm = Llama(
            self.path,
            n_ctx=16000, 
            n_threads=4,
            n_gpu_layers=18,
            n_batch=512,
            n_ubatch=256,
            verbose=False
        )
    
    def __exit__(self, exc_type, exc, tb):
        self.llm.close()
        return True
    
class _ToolSession:
    def __exit__(self, exc_type, exc, tb):
        pass