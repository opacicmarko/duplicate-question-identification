import numpy as np
import pandas as pd
import spacy
import csv
from dataclasses import dataclass
from typing import Counter, Generator, Any, Union

spacy.require_gpu()
nlp = spacy.blank('en')

def preprocess_text(text: str) -> list[str]:
    if type(text) is not str:
        print(f"text parameter is not string, but {type(text)}: {text}")
        return []
    doc = nlp(text)
    return [token.text.lower() for token in doc if not token.is_punct and not token.is_stop and not token.is_space]

@dataclass
class QuestionPair:
    id: int
    qid1: int
    qid2: int
    question1: str
    question2: str
    is_duplicate: bool

@dataclass
class MojaveVocab:
    _words_counter: Counter[str]
    idx_words: dict[str]
    words_idx: dict[str]
    def __init__(self):
        self._words_counter = Counter()
        self.idx_words = None
    def __len__(self):
        return len(self.idx_words)
    def update(self, text: str):
        tokens = preprocess_text(text)
        self._words_counter.update(tokens)
    def finalize(self):
        words_sorted = sorted(self._words_counter.items(), key=lambda x: x[1], reverse=True)
        wordlist_sorted = list(map(lambda x: x[0], words_sorted))
        self.idx_words = { idx: word for idx, word in enumerate(wordlist_sorted) }
        self.words_idx = { word: idx for idx, word in enumerate(wordlist_sorted) }

        self.pad_idx = len(self.idx_words)
        self.unk_idx = self.pad_idx + 1
        
        self.idx_words[self.pad_idx] = '<PAD>'
        self.idx_words[self.unk_idx] = '<UNK>'
        
        self.words_idx['<PAD>'] = self.pad_idx
        self.words_idx['<UNK>'] = self.unk_idx

@dataclass
class ProcessedResult:
    vocab: MojaveVocab
    embedding: np.array
    avg_embedding: np.array

def questions_generator(path: str) -> Generator[dict[str, str], Any, None]:
    with open(path) as questions_file:
        header = questions_file.readline()
        columns = tuple(map(lambda x: x.replace('"', ''), header.split(',')))
        for line in questions_file:
            fields = tuple(map(lambda x: x.replace('"', ''), line.strip().split('","')))
            yield dict(zip(columns, fields))

def question_pairs_generator(path: str) -> Generator[QuestionPair, Any, None]:
    with open(path) as questions_file:
        header = questions_file.readline()
        for line in questions_file:
            fields = list(csv.reader([line]))[0]
            # fields = tuple(map(lambda x: x.replace('"', ''), line.strip().split('","')))
            if len(fields) != 6:
                return
            yield QuestionPair(fields[0], fields[1], fields[2], fields[3], fields[4], fields[5])

def get_max_len(df: pd.DataFrame) -> int:
    max_len = 0
    for _, row in df.iterrows():
        q1_tokens = preprocess_text(row['question1'])
        q2_tokens = preprocess_text(row['question2'])
        max_len = max(max_len, len(q1_tokens), len(q2_tokens))
    return max_len

def load_glove_embeddings(path: str, calculate_average: bool = False) -> Union[dict[str, np.array], tuple[dict[str, np.array], np.array]]:
    embeddings = {}
    vecs = None
    if calculate_average == True:
        vecs = np.zeros((400000, 300), dtype=np.float32)
    with open(path) as glove_file:
        for i, line in enumerate(glove_file):
            values = line.split()
            word = values[0]
            glove_vector = np.array(values[1:])
            if vecs is not None:
                vecs[i] = glove_vector
            embeddings[word] = glove_vector
    if calculate_average == True:
        avg_vec = vecs.mean(axis=0)
        return (embeddings, avg_vec)
    return embeddings

def load_and_embed_questions(questions_path: str, glove_path: str, embeddings=None, avg_embedding=None) -> ProcessedResult:
    print('Loading GloVe embeddings...', end='')
    if embeddings is None:
        embeddings, avg_embedding = load_glove_embeddings(glove_path, calculate_average=True)
    print('DONE')
    print('Generating vocabulary...', end='')
    vocab = MojaveVocab()
    for qpair in question_pairs_generator(questions_path):
        vocab.update(qpair.question1)
        vocab.update(qpair.question2)
    vocab.finalize()
    print('DONE')
    print('Extracting only embeddings required for vocabulary...', end='')

    embeddings_arr = np.zeros((len(vocab), 300), dtype=np.float32)
    for idx, word in vocab.idx_words.items():
        embeddings_arr[idx] = embeddings.get(word, avg_embedding)
    embeddings_arr[vocab.pad_idx] = np.zeros(300)
    embeddings_arr[vocab.unk_idx] = avg_embedding
    print('DONE')
    return ProcessedResult(vocab, embeddings_arr, avg_embedding)