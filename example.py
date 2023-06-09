from llama_cpp import Llama
llm = Llama(model_path="../models/wizardLM-7B.ggmlv3.q4_1.bin")
output = llm(
    "Q: Name the planets in the solar system? A: ", 
    max_tokens=32, 
    stop=["Q:", "\n"], 
    echo=True
    )
print(output)