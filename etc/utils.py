#%%
import json
import numpy as np
import pandas as pd
from cilin import Cilin
from CompoTree import Radicals
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report as cls_report
from collections import Counter
from itertools import product, chain


def classification_report(y_test, predictions):
    return pd.DataFrame(cls_report(y_test, predictions, output_dict=True)).T

def train_test_split(df_feature, df_tgt, tgt_col="lev1", test_size=0.2, random_state=101):
    test_idx = df_tgt.groupby(tgt_col).sample(frac=test_size, random_state=random_state).index
    y_test, y_train = df_tgt.iloc[test_idx], df_tgt.drop(test_idx)
    X_test, X_train = df_feature.iloc[test_idx], df_feature.drop(test_idx)
    return X_train, X_test, list(y_train[tgt_col]), list(y_test[tgt_col])


class RadicalSemanticTagger:

    def __init__(self, all_words) -> None:
        self.radicals = Radicals.load()
        with open('radical_semantic_tag.json', encoding='utf-8') as f:
            self.rad_sem = json.load(f)
        self.features = sorted(set(self.tag_words(all_words)))
        self.features = sorted(self.features, key=lambda x: '_' in x )
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



class DocumentTermMatrix:

    def __init__(self, RSTagger, Cilin, level=3) -> None:
        self.Tagger = RSTagger
        self.Cilin = Cilin
        self.docs = { 
            k:list(v) for k, v in Cilin.category_split(level=level).items() 
        }
        self.doc2id = { doc:i for i, doc in enumerate(self.docs.keys()) }
        self.id2doc = { v:k for k, v in self.doc2id.items() }
        self.features = list(self.Tagger.idx2feat.values())
        self._make_dtm()

    @property
    def np(self):
        return self.tfidf_mat

    @property
    def pd(self):
        return pd.DataFrame(self.tfidf_mat, columns=self.features)

    @property
    def df_tgt(self):
        df_tgt = [ self.Cilin._parse_key(x) for x in self.documents ]
        df_tgt = [ (x[0] + " " + self.Cilin.get_tag(x[0]) , 
                    ''.join(x[:2]) + " " + self.Cilin.get_tag(''.join(x[:2])), 
                    ''.join(x[:3])) 
                    for x in df_tgt ]
        return pd.DataFrame(df_tgt, columns="lev1 lev2 lev3".split())
    
    @property
    def documents(self):
        return list(self.doc2id.keys())

    def _make_dtm(self):
        count_mat = []
        for doc in self.doc2id:
            doc = self.Tagger.encode_doc(self.docs[doc])
            count_mat.append(doc)
        self.count_mat = np.array(count_mat)

        idf = TfidfTransformer().fit(self.count_mat)
        tf_idf_mat = idf.transform(self.count_mat)
        self.tfidf_mat = tf_idf_mat.toarray()
