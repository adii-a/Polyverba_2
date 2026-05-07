import sys
import os

from translation.indictrans2 import load_model

tokenizer, model = load_model("en-indic")

print("Tokenizer class:", type(tokenizer))
print("Model class:", type(model))

text = "Hello, how are you?"

inputs_hi = tokenizer(text, return_tensors="pt")
gen_hi = model.generate(**inputs_hi.to(model.device), forced_bos_token_id=tokenizer.convert_tokens_to_ids("hin_Deva"))
print("Hindi:", tokenizer.batch_decode(gen_hi, skip_special_tokens=True))

inputs_ta = tokenizer(text, return_tensors="pt")
gen_ta = model.generate(**inputs_ta.to(model.device), forced_bos_token_id=tokenizer.convert_tokens_to_ids("tam_Drav"))
print("Tamil:", tokenizer.batch_decode(gen_ta, skip_special_tokens=True))

