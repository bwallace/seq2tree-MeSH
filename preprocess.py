'''
Preprocessing module responsible for mapping abstracts
to sequences of integers coding for tokens. 

Partially attributed to:
https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation.ipynb
'''
import unicodedata
import numpy 
import re
import operator 

SOS_token = 0
EOS_token = 1
UNK_token = 2

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"[0-9]+", "qqq", s)
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r"", s)

    if ("qqq" in s) and ("." in s):
        # basically, an explicit 'fraction'
        # symbol
        s = "qqq.qqq"

    if s == "qqq.qqqqqq.qqq":
        import pdb; pdb.set_trace()
    return s


class Preprocessor:
    def __init__(self, vocab_size=10000, split_char=" ", normalize=True):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK"}
        self.split_char = split_char
        self.n_words = 3 # start at 3 to skip {S/E}OS and UNK
        # note; None is effectively arbitrarily large vocab
        self.vocab_size = vocab_size 
        self.normalize = normalize
        self.dim = None 
      
    def tokenize(self, sentence):
        return sentence.split(self.split_char)

    def tokenize_and_normalize_sentence(self, sentence):
        if not self.normalize: 
            return  self.tokenize(sentence)

        normalized = []
        for w in self.tokenize(sentence):
            w_normed = normalize_string(w) 
            if w_normed != "":
                normalized.append(w_normed)

        return normalized

    def index_words(self, sentence):
        for word in self.tokenize_and_normalize_sentence(sentence):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


    def preprocess(self, all_sentences):
        '''
        Assumes all_sentences is a list of sentences; 
        here we map tokens to indices and, subsequently,
        sort words by frequency.
        '''
        for sent in all_sentences:
            self.index_words(sent)

        # now sort by frequency, keep only the top 
        # vocab_size words
        self.sorted_words = sorted(self.word2count.items(), 
                              key=operator.itemgetter(1),
                              reverse=True)

        # update word indices accordingly; +3
        # accounts for {S/E}OS and qqq
        self.word2index = {}
        for idx, w in list(enumerate(self.sorted_words))[:self.vocab_size]:
            self.word2index[w[0]] = idx + 3

        self.dim = max(self.word2index.values())


    def encode_text(self, text):
        out = [SOS_token]
        for w in self.tokenize_and_normalize_sentence(text):
            if w in self.word2index:
                out.append(self.word2index[w])
            else:
                out.append(UNK_token)
        out.append(EOS_token)
        return out

    def encode_texts(self, texts):
        X = []
        for text in texts:
            X.append(self.encode_text(text))
        return X

    def get_dim(self):
        return self.dim

