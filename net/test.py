from llama_cpp import Llama 
from search import QuerySummarizer 

llm = Llama(
    model_path="../llm/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf", 
    n_ctx=16000, 
    n_threads=4,
    n_gpu_layers=18, # 15 to 20
    n_batch=512,
    n_ubatch=256,
    verbose=False
) 

summarizer = QuerySummarizer(llm) 

query = "Current political relation between commonwealth nations"

print(query)
print(summarizer(query))

llm.close()