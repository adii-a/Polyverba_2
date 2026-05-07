import inspect
from translation.indictrans2 import load_model

tokenizer, _ = load_model("en-indic")
print(inspect.signature(tokenizer.__call__))
