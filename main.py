import json
from typing import List

import guidance
from guidance import models, select, system, user, assistant, gen
from guidance.models import Model
from pydantic import BaseModel

from kg import extract_kg, clean_extracted_kg, get_entities, get_subkg, kg_to_nl, extract_kg_with_subkg, merge_kg
from models import KGList


with open('data/monte.txt', 'r', encoding="utf8") as file:
    book_text = file.read()
book_text = book_text.replace('\n', ' ')
book_text = book_text[51:]

model = models.LlamaCpp('D:\PythonPlayground\llama.cpp-master\models\mistral-7b-instruct-v0.1.Q6_K.gguf', n_ctx=3500) 


chunk_size = 2000
main_kg = None
for count, i in enumerate(rangtranne(0, len(book_text), chunk_size)):
    text = book_text[i:(i+chunk_size)-1]
    if i == 0:    
        kg_ = extract_kg(model, text)
        main_kg = clean_extracted_kg(kg_, timestamp=count+1)
        continue

    ents = get_entities(text)
    sub_kg_ = get_subkg(main_kg, ents)
    kg_ = extract_kg_with_subkg(model, text, sub_kg_)
    clean_kg_ = clean_extracted_kg(kg_, timestamp=count+1)
    main_kg = merge_kg(main_kg, clean_kg_)
