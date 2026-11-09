import struct
import numpy as np
import torch
import torch.nn as nn

RANDOMIZED = False

ANNOTATION_FILE = "/mnt/ssd/flickr30k/results_20130124.token"
WORD2VEC_FILE = "/mnt/ssd/word2vec/GoogleNews-vectors-negative300.bin"

print("Computing vocabulary...")

vocabulary = set()

with open(ANNOTATION_FILE, encoding="utf-8") as f:
    for line in f:
        _, words = line.strip().split("\t", 1)
        for word in words.lower().split():
            vocabulary.add(word)

print("Number of words:", len(vocabulary))


def read_until(open_file):
    while True:
        char = open_file.read(1)

        if not char:
            raise EOFError("Fin inattendue du fichier Word2Vec")

        if char == b" ":
            break

        yield char


def read_word2vec():
    with open(WORD2VEC_FILE, "rb") as f:
        first_line = f.readline()
        n_words, n_dim = map(int, first_line.split())

        for _ in range(n_words):
            word = b"".join(read_until(f)).decode(
                "utf-8", errors="ignore"
            )

            vector = struct.unpack(
                "f" * n_dim,
                f.read(n_dim * 4)
            )

            yield word, vector


def create_embedding_layer(vocab, randomized=False):
    # 0 : padding et marqueur spécial du TP
    idx_to_word = [""]
    word_to_idx = {"": 0}
    weights = [np.zeros(300, dtype=np.float32)]

    if randomized:
        for word in sorted(vocab):
            idx_to_word.append(word)
            word_to_idx[word] = len(idx_to_word) - 1
            weights.append(
                np.random.normal(0, 0.01, 300).astype(np.float32)
            )

    else:
        # Lecture des vecteurs pré-entraînés
        for word, vector in read_word2vec():
            if word in vocab and word not in word_to_idx:
                idx_to_word.append(word)
                word_to_idx[word] = len(idx_to_word) - 1

                weights.append(
                    np.asarray(vector, dtype=np.float32)
                )

    weights = np.stack(weights)

    emb_layer = nn.Embedding.from_pretrained(
        torch.tensor(weights, dtype=torch.float32),
        freeze=not randomized,
        padding_idx=0
    )

    return emb_layer, idx_to_word, word_to_idx


def tokenize(sentence, word2idx, max_length=20):
    res = [0]

    sentence = sentence.lower()

    for word in sentence.split():
        if word in word2idx:
            res.append(word2idx[word])

    if len(res) > max_length:
        return res[:max_length]

    return res + [0] * (max_length - len(res))


def untokenize(tokens, idx2word):
    return [idx2word[int(x)] for x in tokens]


if __name__ == "__main__":
    pretrained_embeddings, idx2word, word2idx = (
        create_embedding_layer(vocabulary, RANDOMIZED)
    )

    print("Vocabulary size:", len(idx2word))
    print("Embedding shape:", pretrained_embeddings.weight.shape)

    sentence = "A dog is running"
    tokens = tokenize(sentence, word2idx)

    print("Phrase originale :", sentence)
    print("Tokens :", tokens)
    print("Mots reconstruits :", untokenize(tokens, idx2word))
