import torch
import torch.nn as nn

class ProjE(nn.Module):
    def __init__(self, nEntity, nRelation, nEmbedding, margin=1):
        super(ProjE, self).__init__()
        self.margin = margin
        self.embedEntity = nn.Embedding(nEntity, nEmbedding)
        self.embedRelation = nn.Embedding(nRelation, nEmbedding)

        self.linear = nn.Linear(nEmbedding, nEmbedding)

    def forward(self, triple):
        headEntity = triple[:, 0]
        relation = triple[:, 1]
        tailEntity = triple[:, 2]
        h = self.embedEntity(headEntity)
        r = self.embedRelation(relation)
        t = self.embedEntity(tailEntity)
        h = self.linear(h)
        r = self.linear(r)
        t = self.linear(t)
        out = h+r-t # self.linear(h+r-t)
        return out

    def EmbDict(self, x):
        return self.embedEntity(x)



# if __name__ == '__main__':
#     dirEntity = "./FB15K/entity2id.txt"
