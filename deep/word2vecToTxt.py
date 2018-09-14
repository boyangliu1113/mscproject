#! /usr/bin/env python
# Convert a .bin format for word2vec to tst
from gensim.models.keyedvectors import KeyedVectors

print("Convert a .bin format for word2vec to txt")
model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
model.save_word2vec_format('./GoogleNews-vectors-negative300.txt', binary=False)