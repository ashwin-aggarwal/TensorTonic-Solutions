import numpy as np

def softmax(x, axis):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
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

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    hidden = np.maximum(0, np.dot(x, W1) + b1)
    return np.dot(hidden, W2) + b2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    attn_out = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    x = layer_norm(x + attn_out, gamma1, beta1)
    ffn_out = feed_forward(x, W1, b1, W2, b2)
    x = layer_norm(x + ffn_out, gamma2, beta2)
    return x




    