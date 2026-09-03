import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    # return(Q.shape)
    *batch, n_q, d_q = Q.shape
    *batch, n_k, d_k = K.shape
    *batch, n_v, d_v = V.shape

    scores = Q @ K.transpose(-2,-1) 
    #switching last two axes, dont want to switch batch dimensions
    scaled_score = scores / math.sqrt(d_k)
    
    W = F.softmax(scaled_score, dim = -1)
    # weights is size n_q x n_k
    #all the ns should be equal
    #V is size 

    output = W @ V
    return output
