import torch
import torch.nn as nn
import numpy as np
import random 
#from sampler import get_pair_list

# class CLoss(nn.Module):
#     def __init__(self):
#         super(CLoss, self).__init__()
#         self.input_size = 130
#         self.hidden_sizes = [60, 20, 1]

#         self.fc1 = nn.Linear(self.input_size, self.hidden_sizes[0])
#         self.fc2= nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])
#         self.fc3 = nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2])

#     def forward(self, x):
#         out = torch.relu_(self.fc1(x))
#         out = torch.relu_(self.fc2(out))
#         out = torch.softmax(self.fc3(out).squeeze(), dim=0)
        
#         # out = torch.softmax(self.fc1(x).squeeze(), dim=1)


#         return out



# def func(embeddings, labels):
#     # resort labels to see pairs next to each other [12, 12, 43, 43, 100, 100, 100, 100, 54, 54, ....] len bacth
#     # resort embeddings accordingly so that corresponding embeddings is next to each other similar to labels
#     # print(labels)
#     pairs = get_pair_list(labels, both_directions=True, n_max_pairs=300)
#     # print("Pairs e.g.", pairs[:10])

#     mini_batch_size = 20

#     batch_idxs, batch_hot = list(), list()
#     for pair in pairs:
#         mini_batch_idxs, mini_batch_hot = list(), list()
#         mini_batch_idxs.extend(pair)
#         mini_batch_hot.extend([1, 1])
#         while len(mini_batch_idxs) < mini_batch_size:
#             rand_idx = np.random.randint(0, len(labels))
#             if pair[0] != labels[rand_idx] and rand_idx not in mini_batch_idxs:
#                 mini_batch_idxs.append(rand_idx) 
#                 mini_batch_hot.append(0)
#         batch_idxs.append(mini_batch_idxs)
#         batch_hot.append(mini_batch_hot)

#     # print("mini batch idxs e.g.", batch_idxs[:10])
#     # print("mini batch hot e.g.", batch_hot[:10])

#     batch_idxs_shuffled, batch_hot_shuffled = tuplets_shuffle(batch_idxs, batch_hot)

#     # print("mini batch idxs shuffled e.g.", batch_idxs_shuffled[:10])
#     # print("mini batch hot shuffled e.g.", batch_hot_shuffled[:10])

#     total_loss = tuplet_loss(torch.tensor(batch_idxs), embeddings)



#     # bce_loss = nn.BCELoss()
#     # total_loss = 0
#     # for tuplet, hot in zip(batch_idxs_shuffled, batch_hot_shuffled):
#     #     x = torch.softmax(embeddings[tuplet], dim=0)
#     #     hot_t = torch.FloatTensor(hot).cuda()
#     #     loss = bce_loss(x, hot_t)
#     #     total_loss += loss

#     # print(total_loss)
#     # # create tuplet from labels
#     #     # segment 2, step 2 from labels and add n negatives
#     # batch_idxs, batch_hot = list(), list()

#     # for c in range(0,len(labels_resorted), 2):
#     #     mini_batch_idxs, mini_batch_hot = list(), list()
#     #     mini_batch_idxs.extend(range(c,c+2,1))
#     #     mini_batch_hot.extend([1, 1])
#     #     # print(mini_batch_idxs)
#     #     while len(mini_batch_idxs) < 5:
#     #             rand_idx = np.random.randint(0, len(embeddings))
#     #             if mini_batch_idxs[0] != labels_resorted[rand_idx] and rand_idx not in mini_batch_idxs:
#     #                 mini_batch_idxs.append(rand_idx) 
#     #                 mini_batch_hot.append(0)
#     #     # print(mini_batch_idxs)
#     #     batch_idxs.append(mini_batch_idxs)
#     #     batch_hot.append(mini_batch_hot)
    

#     # # SHuffle 
#     # batch_idxs, batch_hot = tuplets_shuffle(batch_idxs, batch_hot)

#     # # print(embeddings_resorted.shape)
#     # mini_batch_embeddings = torch.ones((150, 5, 130), device="cuda")

#     # for i in range(150):
#     #     # print(batch_idxs[i])
#     #     mini_batch_embeddings[i] = embeddings_resorted[batch_idxs[i]]
#     # # print(mini_batch_embeddings.shape)
#     # # print(mini_batch_embeddings[0])
#     # closs= CLoss().cuda()
#     # out = closs(mini_batch_embeddings)
#     # # print(out.shape)
#     # # print(out)

#     # # print(batch_hot.shape)
#     # batch_hot_t = torch.tensor(batch_hot, device="cuda", dtype=torch.float32)
#     # # print(batch_hot_t.shape)
#     # # print(batch_hot_t[0])

#     # bce_loss = nn.BCELoss()

#     # loss = bce_loss(out, batch_hot_t)
    
#     # print(loss)
#     return total_loss



def tuplets_shuffle(tuplet_list_idxs, tuplet_list_hot):
    batch_idxs_shuffled, batch_hot_shuffle = list(), list() 
    for x, y in zip(tuplet_list_idxs, tuplet_list_hot):
        temp = list(zip(x, y))
        random.shuffle(temp)
        res1, res2 = zip(*temp)
        batch_idxs_shuffled.append(list(res1))
        batch_hot_shuffle.append(list(res2))

    # print("\nAfter shuffle:")
    # print(batch_idxs_shuffled[0])
    # print(batch_hot_shuffle[0])

    return batch_idxs_shuffled, batch_hot_shuffle
# batch_idxs, batch_hot = list(), list()

# for pair in pair_list:
#     mini_batch_idxs, mini_batch_hot = list(), list()
#     mini_batch_idxs.extend(pair)
#     mini_batch_hot.extend([1, 1])
#     while len(mini_batch_idxs) < 5:
#         rand_idx = np.random.randint(0, n_classes)
#         if pair[0] != labels_np[rand_idx] and rand_idx not in mini_batch_idxs:
#             mini_batch_idxs.append(rand_idx) 
#             mini_batch_hot.append(0)
#     batch_idxs.append(mini_batch_idxs)
#     batch_hot.append(mini_batch_hot)
# def tuplet_loss(tuplet_batch, embeddings):
#     # Indices
#     a = tuplet_batch[:, 0]
#     p = tuplet_batch[:, 1]
#     n = tuplet_batch[:, 2:]
#     # n = torch.tensor(n)
#     # Embeddings
#     anchor = embeddings[a]
#     positive = embeddings[p]
#     negatives = embeddings[n]
#     # print(anchor.shape)
#     # print(negatives.shape)
#     # negative = negatives[:, 0, :]
#     # print(negative.shape)
#     # anchor.matmul(negative.T)

#     summation = sum([torch.exp(torch.diagonal(anchor.matmul(negatives[:,i,:].T) - anchor.matmul(positive.T))) for i in range(18)])
#     # print(summation)
#     loss = torch.log(1 + summation)
    
#     if len(loss[loss<0]) > 0:
#         print("JJJJJ")
#     loss_mean = loss.mean()
#     return loss_mean


# def contrastive_loss(sim, i, j, t=0.1):
#     N = int(sim.shape[0]/2)
#     a = torch.exp(sim[i,j]/t)
#     b = sum([torch.exp(sim[i,k]/t) for k in range(2*N) if k!=i]) 
#     l = -torch.log(a/b)
#     return l

# def cosine_sim(x1, x2=None, eps=1e-8):
#     x2 = x1 if x2 is None else x2
#     w1 = x1.norm(p=2, dim=1, keepdim=True)
#     w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
#     return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

def contrastive_loss(embeddings, t=0.1):
    # emebddings sorted in pairs

    def cosine_sim(x1, x2=None, eps=1e-8):
        x2 = x1 if x2 is None else x2
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
        return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

    sim = cosine_sim(embeddings, embeddings)
    sim_sum = torch.sum(torch.exp(sim/t), dim=1, keepdim=True)

    minibatch_loss = torch.tensor(0, dtype=torch.float32, device=embeddings.get_device())
    N  = int(sim.shape[0]/2)
    for k in range(1,N):
        i = (2*k-1)-1
        j = (2*k) -1 
        a = torch.exp(sim[i,j]/t)
        b = (sim_sum[i] - torch.exp(sim[i,i]/t)) 
        minibatch_loss = minibatch_loss + (-torch.log(a/b))

    loss = minibatch_loss/N
    return loss
