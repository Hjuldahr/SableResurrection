from llama_cpp import Llama 
from search import QuerySummarizer 

llm = Llama(
    model_path="../llm/Meta-Llama-3-8B-Instruct.Q3_K_M.gguf", 
    n_ctx=4096, 
    n_threads=4, 
    n_gpu_layers=16,
    verbose=False
) 

summarizer = QuerySummarizer(llm) 

print(summarizer( "What are the main causes of the Great Depression?" ))