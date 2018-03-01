import os
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def build_dict(self, word, idx):
        self.idx2word.append(word)
        self.word2idx[word] = int(idx)
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.entity2id = self.build_dict(os.path.join(path, 'entity2id.txt'))
        self.relation2id = self.build_dict(os.path.join(path, 'relation2id.txt'))
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        # self.negTriple = self.tokenize(os.path.join(path, 'train_neg.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def build_dict(self, path):
        """Tokenizes a text file."""
        #assert os.path.exists(path)
        # Add words to the dictionary
        dictionary = Dictionary()
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                dictionary.build_dict(words[0], words[1])
        return dictionary


    def tokenize(self, path):
        """Tokenizes a text file."""
        #assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = len(f.readlines())

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens, 3)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                # if words[0] in self.entity2id.word2idx:
                #     ids[token][0] = self.entity2id.word2idx[words[0]]
                # else:
                #     newItem = len(self.entity2id.idx2word)
                #     self.entity2id.idx2word.append(words[0])
                #     self.entity2id.word2idx[words[0]] = newItem

                ids[token][0] = self.entity2id.word2idx[words[0]]
                ids[token][1] = self.relation2id.word2idx[words[1]]
                ids[token][2] = self.entity2id.word2idx[words[2]]
                token += 1
        return ids

# train = data.train
# test = data.test
# train_file = open(os.path.join(path, "trainIdData.txt"), 'w')
# valid_file = open(os.path.join(path, "validIdData.txt"), 'w')
# test_file = open(os.path.join(path, "testIdData.txt"), 'w')
#
#
#
# train_file.close()
# valid_file.close()
# test_file.close()

