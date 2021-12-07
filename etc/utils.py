#%%
import json
import numpy as np
from cilin import Cilin
from CompoTree import Radicals
from collections import Counter
from itertools import product, chain



class RadicalSemanticTagger:

    def __init__(self, all_words) -> None:
        self.radicals = Radicals.load()
        with open('radical_semantic_tag.json', encoding='utf-8') as f:
            self.rad_sem = json.load(f)
        self.features = set(self.tag_words(all_words))
        self.feat2idx = {
            f:i for i, f in enumerate(self.features)
        }
        self.idx2feat = {
            v:k for k, v in self.feat2idx.items()
        }


    def encode_doc(self, words, prob=False):
        vec = np.zeros(len(self.feat2idx), dtype=int)
        for feat, fq in self.bow(words).items():
            i = self.feat2idx[feat]
            vec[i] = fq
        if prob:
            return vec / sum(vec)
        return vec


    def bow(self, words, counter=True):
        tags = self.tag_words(words)
        if counter:
            return Counter(tags)
        else:
            return list(tags)


    def tag_words(self, words:list):
        tags = chain.from_iterable(self.tag_word(w) for w in words)
        return tags #Counter(tags)


    def tag_word(self, word:str):
        tags = self.sem_feats(word)
        if len(word) == 2: return self.feat_comb(tags)
        if len(word) > 2 or len(word) == 1: 
            return list(chain.from_iterable(tags))


    def sem_feats(self, word):
        rads = [ self.radicals.query(c)[0] for c in word ]
        return [ self.rad_sem.get(r, ["NULL"]) for r in rads ]


    def feat_comb(self, feats):
        if len(feats) > 2: 
            raise Exception('Accepts 2 features only')
        if len(feats) == 1: return feats[0]
        f1, f2 = feats
        return [f"{x}_{y}" for x, y in product(f1, f2)]


# C = Cilin(trad=True)
# documents = { k:list(v) for k, v in C.category_split(level=3).items() }

# # %%
# tagger = RadicalSemanticTagger()

# # %%
# tagger.tag_word('我你它')
# # %%
# words = documents['Aa01']
# words
# #%%
# tagger.tag_words(words)
# # %%
