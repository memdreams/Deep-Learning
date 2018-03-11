import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parameter as Parameter
from torch.autograd import Variable

class ProjE(nn.Module):
    def __init__(self, nEntity, nRelation, nEmbedding=100, n_batch=1, dropout=0.5, margin=1):
        super(ProjE, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.margin = margin
        self.embedEntity = nn.Embedding(nEntity, nEmbedding)     # WE
        self.embedRelation = nn.Embedding(nRelation, nEmbedding) # WR

        # Combination Operator
        self.De = Parameter.Parameter(torch.eye(nEmbedding, nEmbedding))
        self.Dr = Parameter.Parameter(torch.eye(nEmbedding, nEmbedding), requires_grad=True)
        self.b_c = Parameter.Parameter(torch.zeros(n_batch, nEmbedding), requires_grad=True)
        # Combination.op = self.embedEntity@self.De +  self.embedRelation@self.Dr + self.b_c

        self.linear1 = nn.Linear(nEmbedding, nEmbedding)
        self.init_weights()

    def forward(self, triple, flag=0): # flag = 0: hr->t; 1: rt->h
        headEntity = triple[:, 0]
        relation = triple[:, 1]
        tailEntity = triple[:, 2]
        h = self.embedEntity(headEntity)
        r = self.embedRelation(relation)
        t = self.embedEntity(tailEntity)
        comb_op = h@self.De + r@self.Dr + self.b_c if flag==0 else t@self.De + r@self.Dr + self.b_c
        # Embedding Projection Function
        f = F.tanh(comb_op)
        h_e_r = Variable(torch.zeros(triple.data.shape[0], 1))
        for i in range(triple.data.shape[0]):
            h_e_r[i] = f[i]@t[i] if flag==0 else f[i]@h[i]

        h_e_r = torch.sigmoid(h_e_r)
        return h_e_r

    def EmbDict(self, x):
        return self.embedEntity(x)

    def init_weights(self):
        self.embedEntity.weight.data.uniform_(-0.1, 0.1)
        self.embedRelation.weight.data.uniform_(-0.1, 0.1)




# if __name__ == '__main__':
#     dirEntity = "./FB15K/entity2id.txt"
