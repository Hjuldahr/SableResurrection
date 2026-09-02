from llama_cpp.llama_chat_format import LlamaChatCompletionHandlerRegistry

# Get a sorted list of all registered chat format strings
formats = sorted(list(LlamaChatCompletionHandlerRegistry._chat_handlers.keys()))
print(formats)