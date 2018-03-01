import os

path = "./FB15K-237"
train_rawdata = "train.txt"
test_rawdata = "test.txt"
file = open(os.path.join(path, "train.txt"))

# file.read()
entityF = open(os.path.join(path, "entity2id.txt"), 'w')
relationF = open(os.path.join(path, "relation2id.txt"), 'w')

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, item, filename):
        if item not in self.word2idx:
            self.word2idx[item] = len(self.idx2word)
            self.idx2word.append(item)
            filename.write(item)
            filename.write('\t')
            filename.write(str(self.word2idx[item]))
            filename.write('\n')
        return self.word2idx[item]

    def __len__(self):
        return len(self.idx2word)

class Transform(object):
    def __init__(self, path):
        self.entitiesDic = Dictionary()
        self.relationsDic = Dictionary()
        self.traindata = os.path.join(path, train_rawdata)

    def transformData(self, path, batch_size=20):
        assert os.path.exists(path)
        # Add entities and relationship to the dictionary
        with open(path, "r") as f:
            tokens = 0
            for line in f:
                words = line.split()
                self.entitiesDic.add_word(words[0], entityF)
                self.entitiesDic.add_word(words[2], entityF)
                self.relationsDic.add_word(words[1], relationF)
                # for entity in (self.entitiesDic.idx2word[-2:]):
                #     # entity = self.entitiesDic.idx2word
                #     entityF.write(entity)
                #     entityF.write('\t')
                #     entityF.write(str(self.entitiesDic.word2idx[entity]))
                #     entityF.write('\n')
                # relation = self.relationsDic.idx2word[-1]
                # relationF.write(relation)
                # relationF.write('\t')
                # relationF.write(str(self.relationsDic.word2idx[relation]))
                # relationF.write('\n')


entitiesDic = Dictionary()
relationsDic = Dictionary()
# entityF = open(os.path.join(path, "entity2id.txt"), 'a')
# relationF = open(os.path.join(path, "relation2id.txt"), 'a')

# for line in file:
#     words = line.split()
#     entities = words[0]
#     entitiesDic.add_word(words[0])
#     entitiesDic.add_word(words[2])
#     relationsDic.add_word(words[1])
#     for i in range(len(entitiesDic.idx2word)):
#         entity = entitiesDic.idx2word
#         entityF.write(entity[i])
#         entityF.write(str(entitiesDic.word2idx[entity[i]]))

transform = Transform(path)
traindatapath = os.path.join(path, train_rawdata)
transform.transformData(traindatapath)

entityF.close()
relationF.close()
    # print(words)
