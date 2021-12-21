#%%
import json
import numpy as np
import pandas as pd
from cilin import Cilin
from CompoTree import Radicals
from collections import Counter
from itertools import product, chain
from random import sample
from utils import *


def all_words():
    return chain.from_iterable(C.category_split().values())

def intersect(wt, rt):
    s1 = C.get_members(wt)
    s2 = Tagger.tag2words.get(rt)
    return s2.intersection(s1)

C = Cilin(trad=True)
Tagger = RadicalSemanticTagger(all_words=all_words(), bigram=True, word_type="single")
# s1 = Tagger.tag2words.get('無生命_住宿')  # G: 無生命_人體精神  C: 人體四肢_無生命, 無生命_住宿   # F: 人體四肢_人體頭部, 城鄉_人體頭部 (drastic semantic change)



# %%
