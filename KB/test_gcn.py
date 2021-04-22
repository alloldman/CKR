import torch.sparse as sparse
import torch
i = torch.LongTensor([[0, 1, 1],
                      [2, 0, 2]])
v = torch.FloatTensor([3, 4, 5])
x1 = sparse.FloatTensor(i, v)
x2= sparse.FloatTensor(i, v)
x = torch.sparse.sum(x1,dim=1)

print(x1.mm(x2.t().to_dense()))
print(x1.mul(x.to_dense().reshape(-1,1)))