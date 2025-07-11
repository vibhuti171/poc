#just an example
import transformers
import torch

model_id = "nvidia/OpenCodeReasoning-Nemotron-1.1-32B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

prompt = """You are a helpful and harmless assistant. You should think step-by-step before responding to the instruction below.

Please use python programming language only.

You must use ```python for just the final solution code block with the following format:
```python
# Your code here
```

{user}
"""

messages = [
    {
        "role": "user",
        "content": prompt.format(user="Write a program to calculate the sum of the first $N$ fibonacci numbers")
    },
]

outputs = pipeline(
    messages,
    max_new_tokens=49152,
)
print(outputs[0]["generated_text"][-1]['content'])
