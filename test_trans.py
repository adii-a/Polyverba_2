import sys
import os

from translation.indictrans2 import translate_text, load_model

print("Translating to Hindi:")
print(translate_text("Hello, how are you?", src="eng_Latn", tgt="hin_Deva"))

print("Translating to Tamil:")
print(translate_text("Hello, how are you?", src="eng_Latn", tgt="tam_Drav"))

