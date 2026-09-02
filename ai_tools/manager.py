from llama_cpp import Llama

from ai_tools.calculate import calculate
from ai_tools.clock import clock
from ai_tools.file_io import FileHandler, browse_file_candidates, delete_files, write_file
from ai_tools.web_search_v2 import PageSummarizer, search_books, search_image, search_news, search_text

class ToolManager:
    def __init__(self, path: str | None = None):
        self.path = path or 'C:\\Users\\robert\\Documents\\VS Code Files\\SABLE-Revamp\\llm\\gemma-4-E2B-it-Q4_K_M.gguf'
        
        self._llm: Llama | None = None
        self._page_summarizer: PageSummarizer | None = None
        self._file_handler: FileHandler | None = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc, tb):
        if not self._llm:
            return False
            
        self._page_summarizer = None
        self._file_handler = None
        
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
    
    def clock(self, zone: str | None = None) -> str:
        return clock(zone)
    
    def calculate(self, expression: str) -> str:
        return calculate(expression)
    
    def search_text(self, query: str, mode: str = "text") -> str:
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

SEARCH_TEXT_META = {
    "type": "function",
    "function": {
        "name": "search_text",
        "description": (
            "Get a listing of up to 10 textual web results from the internet using a query. "
            "Use the text mode for general web results, books for bibliographic metadata, "
            "or news for publication metadata."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query used to search the internet.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["text", "books", "news"],
                    "description": "The type of search results to return. Defaults to text.",
                },
            },
            "required": ["query"],
        },
    },
}

SEARCH_BOOKS_META = {
    "type": "function",
    "function": {
        "name": "search_books",
        "description":(
            "Get a listing of up to 10 books from the internet using a query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query being used to search the internet",
                }
            },
            "required": ["query"],
        },
    },
}

SUMMARIZE_PAGE_META = {
    "type": "function",
    "function": {
        "name": "summarize_page",
        "description":(
            "Get a query oriented summary of the pages textual contents assuming its correctly formatted. "
            "This is not a comprehensive breakdown of the page's markdown contents and its link structure, "
            "but an AI generated informative paragraph addressing the query directly."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query being used to refine the summary",
                },
                "url": {
                    "type": "string",
                    "description": "The url of the page being summarized",
                },
            },
            "required": ["query", "url"],
        },
    },
}

SUMMARIZE_FILES_META = {
    "type": "function",
    "function": {
        "name": "summarize_files",
        "description": (
            "Summarize up to 10 specified files from the file workspace. "
            "This is not a comprehensive breakdown of the files, "
            "but an AI generated informative paragraph of their collective content. "
            "If a deeper view is needed, you can try reading fewer files at once. "
            "summarize_files can only access files in the file-workspace/ directory."
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

# Llama Tool Schemas
READ_FILE_META = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "Read the contents of a single file from the file workspace. "
            "Which can have its contents sliced using offset and limit. "
            "If limit is negative (-1) it will read up to the end of file. "
            "If you receive fewer lines then limit, the slice either hit the end of file, "
            "Or exceeded the allowed token budget (10,000). "
            "read_file can only access files in the file-workspace/ directory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The filename to use. Must end in a supported non-binary format (.txt, .md, .html, .json, .csv, .py, .js, or .css).",
                    "pattern": "^.*\\.(txt|md|html|json|csv|py|js|css)$"
                },
                "offset": {
                    "type": "integer",
                    "description": "The number of lines to skip. Defaults to the start of the file.",
                    "default": 0
                },
                "limit": {
                    "type": "integer",
                    "description": "The maximum number of lines to return. A negative value reads until end of file or the token budget is reached.",
                    "default": -1
                }
            },
            "required": ["filename"]
        }
    }
}

WRITE_FILE_META = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": (
            "Write or append to a single file in the file workspace. "
            "Newlines need to be added manually for multiline content. "
            "During write mode, file_content will overwrite the file, or clear it if file_content is an empty string. "
            "During append mode, file_content will be added to the end of the file, or do nothing if file_content is an empty string. "
            "If the file does not exist, it will be created. "
            "write_file can only access files in the file-workspace/generated/ directory."
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

DELETE_FILES_META = {
    "type": "function",
    "function": {
        "name": "delete_files",
        "description": (
            "Delete one or more files from the file workspace. "
            "delete_files can only access files in the file-workspace/generated/ directory."
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
                    "maxItems": 10 # artificial limit to reduce file-IO within a single tool loop cycle
                }
            },
            "required": ["filenames"]
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