from llama_cpp import Llama
from pathlib import Path
import os

model_path = Path(os.path.expanduser("~/Library/Application Support/loqmi/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf"))

print(f"File size: {model_path.stat().st_size / (1024*1024*1024):.2f} GB")
print("Initializing model...")

# Bare minimum test
llm = Llama(model_path=str(model_path), n_ctx=2048, verbose=True)

print("Success!")