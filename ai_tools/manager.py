from pathlib import Path

from llama_cpp import Llama

from ai_tools.calculate import calculate
from ai_tools.clock import clock
from disk.file_io import FileHandler
from net.search import QuerySummarizer

class ToolManager:
    def __init__(self, path: str):
        self.path = path
        
        self.llm: Llama | None = None
        self.query_summarizer: QuerySummarizer | None = None
        self.file_handler: FileHandler | None = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc, tb):
        self._clean_model()
        return False
    
    def _build_model(self):
        if self.llm:
            return
        
        self.llm = Llama(
            self.path, 
            n_ctx=16000, 
            n_threads=4,
            n_gpu_layers=18,
            n_batch=512,
            n_ubatch=256,
            verbose=False
        )
        
        self.query_summarizer = QuerySummarizer(self.llm)
        self.file_handler = FileHandler(self.llm)
    
    def _clean_model(self):
        if not self.llm:
            return
            
        self.query_summarizer = None
        self.file_handler = None
        self.llm.close()
        self.llm = None
            
    def clock(self, zone: str | None = None) -> str:
        return clock(zone)
    
    def calculate(self, expression: str) -> str:
        return calculate(expression)
    
    def web_search(self, query: str) -> str:
        self._build_model()
        return self.query_summarizer(query)
    
    def read_file(self, file_ids: list[str]) -> str:
        self._build_model()
        return self.file_handler.read_files(
            *[Path(file_id) for file_id in file_ids]
        )