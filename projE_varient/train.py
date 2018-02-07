import numpy
import torch
import random
import torch.nn.functional as F
from torch.autograd import Variable
import representData as data
import projE_Variant
import torch.utils.data as D

dataset = './FB15K-237/'
train_path = './FB15K-237/train.txt'
corpus = data.Corpus(dataset)

test_len = 2000
ids = corpus.train[0:test_len]
pos = corpus.train
neg = corpus.negTriple
trainLength = len(corpus.train) + len(corpus.negTriple)
nEntity = len(corpus.entity2id)  # entity and relation size
nRelation = len(corpus.relation2id)  # entity and relation size
entityDic = corpus.entity2id

# Hyper Parameters
nEmbed = 20
eval_batch_size = 10
batch_size = 20
train_data = corpus.train
val_data = corpus.valid
test_data = corpus.test
epochs = 5
num_samples = 1000
clip = 5
margin = 1
save = 'model.sv'


model = projE_Variant.ProjE(nEntity, nRelation, nEmbed)

# loss_func = nn.MSELoss()
def loss_func(data):
    loss = torch.sum(data*data, 1)/2/data.data.shape[1]
    return loss
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
            # pos_inputs = Variable(pos[i:i+batch_size])

            # pos_head = triple[:, 0]
            # relation = triple[:, 1]
            # pos_tail = triple[:, 2]
            neg_triples = pickNegTriple(triple)
            pos_triples = Variable(triple)
            neg_triples = Variable(neg_triples)

            # Forward + backword + optimize
            model.zero_grad()
            pos_outputs = model(pos_triples)
            neg_outputs = model(neg_triples)
            loss_pos = loss_func(pos_outputs)
            loss_neg = loss_func(neg_outputs)

            loss = margin + loss_pos - loss_neg
            loss = F.relu(loss)
            loss = torch.sum(loss)
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)
            optimizer.step()
            total_loss += loss.data

            interval = (i+1) //batch_size

            if interval % 100 == 0:
                print('Epoch [%d/%d], Step[%d/%d], Loss: %.3f, Total Loss: %.4f' %
                      (epoch + 1, epochs, interval, batch_size, loss.data[0], total_loss[0]))

    torch.save(model.state_dict(), 'transE_params_65.pkl')   # 只保存网络中的参数 (速度快, 占内存少)

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


# test_loss = test(test_data)
# train()
# test(test_data)
def restore_param():
    model.load_state_dict(torch.load('prijE_params_60.pkl'))
    train()
    test(test_data)
