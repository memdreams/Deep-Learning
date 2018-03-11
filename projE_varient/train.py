import numpy
import os
import torch
import random
import torch.nn.functional as F
from torch.autograd import Variable
import representData as data
import projE_Variant
import torch.utils.data as D

data_dir = './FB15K-237/'
train_path = './FB15K-237/train.txt'
corpus = data.Corpus(data_dir)

# test_len = 2000
# pos = corpus.train[0:test_len]
pos = corpus.train
# neg = corpus.negTriple
trainLength = len(corpus.train) # + len(corpus.negTriple)
nEntity = len(corpus.entity2id)  # entity and relation size
nRelation = len(corpus.relation2id)  # entity and relation size
entityDic = corpus.entity2id
isHeadFlag = 0


def gen_hr_t(triple_data):
    hr_t = dict()
    for h, t, r in triple_data:
        if h not in hr_t:
            hr_t[h] = dict()
        if r not in hr_t[h]:
            hr_t[h][r] = set()
        hr_t[h][r].add(t)

    return hr_t


def gen_tr_h(triple_data):
    tr_h = dict()
    for h, t, r in triple_data:
        if t not in tr_h:
            tr_h[t] = dict()
        if r not in tr_h[t]:
            tr_h[t][r] = set()
        tr_h[t][r].add(h)
    return tr_h

# Hyper Parameters
nEmbed = 20     # 100
eval_batch_size = 10
batch_size = 30  # 20
train_data = corpus.train
val_data = corpus.valid
test_data = corpus.test
epochs = 25
num_samples = 1000
clip = 5
margin = 1
save = 'model.sv'


model = projE_Variant.ProjE(nEntity, nRelation, nEmbed)

# loss_func = nn.MSELoss()
def loss_func(data):
    loss = torch.sum(torch.log(data))
    return loss

lossFunc = torch.nn.CrossEntropyLoss()
lossFunc = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def pickNegTriple(posTriple):
    negTriple = torch.ones(posTriple.shape).type(torch.LongTensor)
    for i, triple in enumerate(posTriple):
        index = random.randint(0, nEntity-1)
        headOrTail = random.choice([0, 1]) # if 1, change head; if 0, change tail
        negTriple[i] = triple
        if headOrTail:
            if index != triple[0]:
                negTriple[i, 0] = index
            else:
                negTriple[i, 0] = (index + 1) % nEntity
        else:
            if index != triple[2]:
                negTriple[i, 2] = index
            else:
                negTriple[i, 2] = (index + 1) % nEntity
    return negTriple



###############################################################################
# Training code
###############################################################################
def train():
    for epoch in range(epochs):
        train_loader = D.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        total_loss = 0
        # for i in range(0, trainLength, batch_size):
        for i, triple in enumerate(train_loader):
            # loss = 0
            #Get batch inputs and targets
            targets = Variable(torch.cat((torch.ones(triple.shape[0], 1), torch.zeros(triple.shape[0], 1))))

            neg_triples = pickNegTriple(triple)
            # output_triples = torch.cat((triple, neg_triples), 0)
            # triples = Variable(output_triples)
            # outputs = model(triples, flag=isHeadFlag)
            # loss = lossFunc(outputs, targets)

            pos_triples = Variable(triple)
            neg_triples = Variable(neg_triples)

            # Forward + backword + optimize
            model.zero_grad()
            pos_outputs = model(pos_triples)
            neg_outputs = model(neg_triples)
            loss_pos = loss_func(pos_outputs)
            loss_neg = loss_func(1-neg_outputs)


            loss = -loss_pos - loss_neg
            # loss = F.relu(loss)
            # loss = torch.sum(loss)
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)
            optimizer.step()
            total_loss += loss.data

            interval = (i+1) //batch_size

            if interval % 100 == 0:
                print('Epoch [%d/%d], Step[%d/%d], Loss: %.3f, Total Loss: %.4f' %
                      (epoch + 1, epochs, interval, batch_size, loss.data[0], total_loss[0]))

    torch.save(model.state_dict(), 'projE_params_50.pkl')   # 只保存网络中的参数 (速度快, 占内存少)

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    valid_loader = D.DataLoader(dataset=data_source, batch_size=batch_size, shuffle=True)

    for i, triple in enumerate(valid_loader):
        data = Variable(triple)
        outputs = model(data)
        # output_flat = output.view(-1, ntokens)
        loss = torch.norm(outputs[0])
        for l in range(1, batch_size):
            loss = loss + torch.norm(outputs[l])
        total_loss += loss.data
        # hidden = repackage_hidden(hidden)
        print("Total Loss: ", total_loss/len(data_source))

    return total_loss[0] / len(data_source)

def test(testData):
    model.eval()
    test_loader = D.DataLoader(dataset=testData, batch_size=5, shuffle=True)
    avg_rank = 0
    mrr = 0.0
    hit10 = 0
    rankdict = {}
    for i, triples in enumerate(test_loader):
        triple = triples[0]
        triple = triple.view(1, triples.shape[1])
        for t in range(0, nEntity - 1):
            rankdict[t] = torch.norm(model(Variable(
                torch.cat((triple[:, 0], triple[:, 1], torch.LongTensor([t])), 0).view(triple.shape[0], triple.shape[1])))).data.numpy()[0]
        ranklist = sorted(rankdict.items(), key=lambda rankdict: rankdict[1], reverse=False)

        for j, row in enumerate(ranklist):
            if row[0] == triple[0][2]:
                avg_rank += j
                if j <= 10:
                    hit10 += 1
                if j > 0:
                    mrr += 1.0 / j
                break

        if i % 10 == 0:
            print("Current raw rank:----")
            print(avg_rank / (i + 1))
            print("Current MRR raw:----")
            print(mrr / (i + 1))
            print("Hit 10 (Raw) per 10 test triples:----")
            print(hit10 / (i + 1))
        #
        if i % 100 == 0:
            print("Hit 10 (Raw) per 100 test triples:----")
            print(hit10 / (i + 1))



# Loop over epochs.
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
# try:
#     for epoch in range(1, epochs+1):
#         # train()
#         val_loss = evaluate(val_data)
#         print('-' * 89)
#         print('| end of epoch {:3d} | valid loss {:5.2f} | '
#                 'valid ppl {:8.2f}'.format(epoch, val_loss))
#         print('-' * 89)
#         # Save the model if the validation loss is the best we've seen so far.
#         if not best_val_loss or val_loss < best_val_loss:
#             with open(save, 'wb') as f:
#                 torch.save(model, f)
#
# except KeyboardInterrupt:
#     print('-' * 89)
#     print('Exiting from training early')

# Load the best saved model.
# with open(save, 'rb') as f:
#     model = torch.load(f)

# Run on test data.
# valid_loss = evaluate(val_data)
#
#
# test_loss = test(test_data)
train()
test(test_data)

def restore_param():
    model.load_state_dict(torch.load('projE_params_25.pkl'))
    train()
    test(test_data)

# restore_param()