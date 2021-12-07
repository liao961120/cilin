import json
from cilin import Cilin
from CompoTree import Radicals
from collections import Counter
from itertools import product, chain


C = Cilin(trad=True)
radicals = Radicals.load()
documents = { k:list(v) for k, v in C.category_split(level=3).items() }

with open('radical_semantic_tag.json', encoding='utf-8') as f:
    rad_sem = json.load(f)


# Tag characters with radical semantic tags
def sem_feats(word):
    rads = [ radicals.query(c)[0] for c in word ]
    return [ rad_sem.get(r, ["NULL"]) for r in rads ]

def feat_comb(feats):
    if len(feats) > 2: 
        raise Exception('Accepts 2 features only')
    if len(feats) == 1: return feats[0]
    f1, f2 = feats
    return [f"{x}_{y}" for x, y in product(f1, f2)]
# feat_comb(sem_feats('屍體'))
# sem_feats("屍")

# Index data for LDA
sem_unigrams = set()
sem_bigrams = set()
unigram_sem = dict()
bigram_sem = dict()
for word in chain.from_iterable(documents.values()):
    if len(word) == 2:
        sem = feat_comb(sem_feats(word))
        for fc in sem:
            sem_bigrams.add(fc)
            bigram_sem.setdefault(word, set()).add(fc)
    for ch in word:
        sem = sem_feats(ch)[0]
        for fc in sem:
            sem_unigrams.add(fc)
            unigram_sem.setdefault(ch, set()).add(fc)

id2doc = list(documents.keys())
doc2id = {doc:i for i, doc in enumerate(id2doc)}

id2unigramSem = list(sem_unigrams)
unigramSem2id = {v:i for i, v in enumerate(id2unigramSem)}

id2bigramSem = list(sem_bigrams)
bigramSem2id = {v:i for i, v in enumerate(id2bigramSem)}

# Indicies for a single document-term matrix
id2semTags = list(sem_unigrams) + list(sem_bigrams)
semTags2id = {v:i for i, v in enumerate(id2semTags)}

# LDA: Encode document as bag-of-words
def encode_doc(doc, unigram):
    data = []
    if unigram is None:
        semTags = [ tag for ch in chain.from_iterable(documents[doc]) for tag in unigram_sem[ch] ]
        semTags2 = [ tag for word in documents[doc] if len(word) == 2 for tag in bigram_sem[word] ]
        for tag, fq in Counter(semTags + semTags2).items():
            col_idx = semTags2id[tag]
            data.append( (col_idx, fq) )
        return data
    if unigram:
        semTags = [ tag for ch in chain.from_iterable(documents[doc]) for tag in unigram_sem[ch] ]
        for tag, fq in Counter(semTags).items():
            col_idx = unigramSem2id[tag]
            data.append( (col_idx, fq) )
        return data
    else:
        semTags = [ tag for word in documents[doc] if len(word) == 2 for tag in bigram_sem[word] ]
        for tag, fq in Counter(semTags).items():
            col_idx = bigramSem2id[tag]
            data.append( (col_idx, fq) )
        return data

def get_document_topic_vec(doc, lda_model, unigram):
    bow = encode_doc(doc, unigram=unigram)
    tp_distr = dict(lda_model.get_document_topics(bow, minimum_probability=0))
    return [ tp_distr.get(i, 0.0) for i in range(lda_model.num_topics)] 