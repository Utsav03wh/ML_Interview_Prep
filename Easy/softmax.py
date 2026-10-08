import torch 


# numerically stavle softmax and log-softmax 
# Keep Dim = true
# The major advantage of keepdim=True shows up when you want to use the output
#  to perform further operations on your original tensor (like subtracting the mean to normalize your data):
# This WORKS perfectly because [2, 3] and [2, 1] are compatible for broadcasting
# x=torch.tensor([1,2,3])
# normalized = x - torch.mean(x, dim=1, keepdim=True)


# # This CRASHES with a RuntimeError: "The size of tensor a (3) must match 
# # the size of tensor b (2) at non-singleton dimension 1"
# normalized = x - torch.mean(x, dim=1, keepdim=False) 


def stable_soft_max(x,dim):

    max_val=torch.max(x, dim, keepdim=True)

    exp=torch.exp(x-max_val)

    return exp/torch.sum(exp,dim,keepdim=True) 


def log_softmax(x,dim):

    max_val=torch.max(x,dim,keepdim=True)

    shifted_x = x-max_val

    log_sum = torch.log(torch.sum(torch.exp(shifted_x,dim),dim,keepdim=True))

    return shifted_x - log_sum 




