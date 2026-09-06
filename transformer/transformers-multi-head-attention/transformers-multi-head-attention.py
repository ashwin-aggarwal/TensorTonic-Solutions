import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here
    # get sizes
    batch, seq_length, d = Q.shape
    d_k = d // num_heads

    # project Q,K,V
    Q = np.dot(Q, W_q)
    K = np.dot(K, W_k)
    V = np.dot(V, W_v)
    
  
    
    def _reshape(X):
        return X.reshape(batch, seq_length, num_heads, d_k).transpose(0,2,1,3)
    Q = _reshape(Q)
    K = _reshape(K)
    V = _reshape(V)
    # scale
    scores = Q @ K.transpose(0,1, 3, 2)
    scores = scores / np.sqrt(d_k)
    w = softmax(scores, axis = -1)
    output = w @ V

    output = output.transpose(0,2,1,3).reshape(batch, seq_length, d)
        
    #concat
    return np.dot(output, W_o)

    #output projection
    