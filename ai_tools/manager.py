from pathlib import Path

from llama_cpp import Llama

from ai_tools.calculate import calculate
from ai_tools.clock import clock
from ai_tools.db import NoteKeeper
from ai_tools.file_io import FileHandler, browse_file_candidates, delete_files, write_file
from ai_tools.web_search_v2 import PageSummarizer, search_books, search_news, search_text

# You can disable tools by adding _ to them but thats a temp testing-only trick since it desyncs the schema advertisement from the implementation

class ToolManager:
    def __init__(self, path: Path):
        self.path = str(path)
        
        self._llm: Llama | None = None
        self._page_summarizer: PageSummarizer | None = None
        self._file_handler: FileHandler | None = None
        self._note_keeper: NoteKeeper | None = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc, tb):
        self._page_summarizer = None
        
        self._file_handler = None
        
        if self._note_keeper:
            self._note_keeper.close()
            self._note_keeper = None
        
        if self._llm:
            self._llm.close()
            self._llm = None
        
        return False
    
    @property
    def llm(self) -> Llama:
        if not self._llm:
            self._llm = Llama(
                self.path, 
                n_ctx=16_384, 
                n_threads=4,
                n_gpu_layers=-1, 
                n_batch=512,
                n_ubatch=256,
                flash_attn=True,
                verbose=False
            )
            
        return self._llm
    
    @property
    def page_summarizer(self) -> PageSummarizer:
        if not self._page_summarizer:
            self._page_summarizer = PageSummarizer(self.llm)
            
        return self._page_summarizer
    
    @property
    def file_handler(self) -> FileHandler:
        if not self._file_handler:
            self._file_handler = FileHandler(self.llm)
            
        return self._file_handler
    
    @property
    def note_keeper(self) -> NoteKeeper:
        if not self._note_keeper:
            self._note_keeper = NoteKeeper('../db')
            
        return self._note_keeper
    
    def execute(self, command: str, **options: str) -> str:
        if command is None:
            return f"WARNING: no command name was provided."
        
        if command.startswith('_'):
            return f"WARNING: '{command}' is not a legal tool."
        
        try:
            method = self.__getattribute__(command)
        except AttributeError:
            return f"WARNING: '{command}' is not a defined tool."

        try:
            return method(**options)
        except TypeError as e:
            return f"ERROR: Invalid arguments for tool '{command}'. Details: {e}"
        except Exception:
            # If this hits, the tool needs to be hardened or schema revised to realign with the protocol
            return f"ERROR: Execution of '{command}' failed."
    
    @staticmethod
    def clock(zone: str | None = None) -> str:
        return clock(zone)
    
    @staticmethod
    def calculate(expression: str) -> str:
        return calculate(expression)
    
    @staticmethod
    def search_text(query: str, mode: str = "text") -> str:
        match mode:
            case "books":
                return search_books(query)
            case "news":
                return search_news(query)
            case _:
                return search_text(query)
    
    def summarize_page(self, query: str, url: str) -> str:
        return self.page_summarizer.summarize_page(query, url)
    
    def summarize_files(self, filenames: list[str]) -> str:
        return self.file_handler.summarize_files(*filenames)
        
    def read_file(self, filename: str, offset: int = 0, limit: int = -1) -> str:
        return self.file_handler.read_file(filename, offset, limit)
        
    def write_file(self, filename: str, file_content: str="", append: bool=True) -> str:
        return write_file(filename, file_content, append)
    
    def delete_files(self, filenames: list[str]) -> str:
        return delete_files(*filenames)
    
    def browse_files(self, pattern: str = '*') -> str:
        return browse_file_candidates(pattern)
    
    def upsert_note(self, topic: str, note: str) -> str:
        return self.note_keeper.upsert_note(topic, note)
    
    def search_topics(self, query: str, limit: int = 10) -> str:
        return self.note_keeper.search_topics(query, limit)
    
    def select_note(self, topic: str) -> str:
        return self.note_keeper.select_note(topic)
    
    def delete_note(self, topic: str) -> str:
        return self.note_keeper.delete_note(topic)