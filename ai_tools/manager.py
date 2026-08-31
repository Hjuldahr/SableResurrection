from pathlib import Path

from llama_cpp import Llama

from ai_tools.calculate import calculate
from ai_tools.clock import clock
from disk.file_io import FileHandler, browse_file_candidates, write_file
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
    
    def read_file(self, filenames: list[str]) -> str:
        self._build_model()
        return self.file_handler.read_files(
            *[Path(filename) for filename in filenames]
        )
        
    def write_file(self, filename: str, file_content: str="", append: bool=True) -> str:
        return write_file(filename, file_content, append)
    
    def browse_files(self, pattern: str = '*') -> str:
        return browse_file_candidates(pattern)
    

CLOCK_META = {
    "type": "function",
    "function": {
    "name": "clock",
        "description": 
            "Get the current system date and time. Use this whenever you need to know the time, date, day of the week, or make relative time references like 'tomorrow' or 'next week'.",
        "parameters": {
            "type": "object",
            "properties": {
                "zone": {
                    "type": "string",
                    "description": "The IANA time zone name string (e.g., 'America/New_York', 'Europe/London', 'Asia/Tokyo'). If not specified or unknown, leave this empty to default to UTC."
                }
            }
        }
    }
}

CALCULATE_META = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description":(
            "Evaluate a mathematical expression and return the exact result. "
            "Use this for arithmetic, numerical calculations, and unit-free "
            "mathematical expressions. Use this tool instead of calculating "
            "results mentally. Prefer standard mathematical functions when "
            "available rather than recreating them from more primitive operations. "
            "For example, use sqrt(x) rather than x**0.5 and hypot(a, b) rather "
            "than sqrt(a**2 + b**2). Mathematical functions such as sin, sqrt, "
            "cbrt, and hypot use standard function notation with values or inner "
            "expressions inside parentheses."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A mathematical expression to evaluate.",
                },
            },
            "required": ["expression"],
        },
    },
}

WEBSEARCH_META = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description":(
            "Search the internet and get a summary of the results. "
            "This is not a comprehensive breakdown of the topic, "
            "but an AI generated informative paragraph addressing the query directly. "
            "If more details are needed, you can use the current summary to refine your next query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A query to search the web with.",
                },
            },
            "required": ["query"],
        },
    },
}

READ_FILE_META = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "Search the file workspace and get a combined summary from up to 10 files. "
            "This is not a comprehensive breakdown of the files, "
            "but an AI generated informative paragraph of their collective content. "
            "If a deeper view is needed, you can try reading fewer files at once."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filenames": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "A sequence of filenames (1 to 10 files).",
                    "minItems": 1,
                    "maxItems": 10
                }
            },
            "required": ["filenames"]
        }
    }
}

WRITE_FILE_META = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": (
            "Write or append to a single file in the file workspace. "
            "During write mode, file_content will overwrite the file, or clear it if file_content is an empty string. "
            "During append mode, file_content will be added to the end of the file, or do nothing if file_content is an empty string. "
            "If the file does not exist, it will be created. write_file can only access files in the file-workspace/generated/ directory. "
            "newlines need to be added manually for multiline content."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The filename to use. Must end in a supported non-binary format (.txt, .md, .html, .json, .csv, .py, .js, or .css).",
                    "pattern": "^.*\\.(txt|md|html|json|csv|py|js|css)$"
                },
                "file_content": {
                    "type": "string",
                    "description": "The file content being written or appended. Pass an empty string to clear the file in write mode. Defaults to an empty string.",
                    "default": ""
                },
                "append": {
                    "type": "boolean",
                    "description": "Set to true for append mode. Set to false for overwrite/write mode. Defaults to true.",
                    "default": True
                }
            },
            "required": ["filename"]
        }
    }
}

BROWSE_FILES_META = {
    "type": "function",
    "function": {
        "name": "browse_files",
        "description": (
            "Using the content of the file workspace, get a list of file metadata that matches the recursive glob pattern. "
            "You will receive: relative file location; approximate file mime type; file size in a human readable format; creation, modification, and access times; and a preview containing up to 256 characters from the first line. "
            "This will only return utf-8 encoded readable files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The recursive glob pattern to use. Defaults to '*'.",
                    "default": "*"
                }
            }
        }
    }
}