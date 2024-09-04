import numpy as np

class AttentionLayer:
    def __init__(self, d_k, d_v):
        self.d_k = d_k
        self.d_v = d_v

    def softmax(self, x, axis=None):
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def forward(self, query, key, value):
        """
        前向传播

        输入:
        - query: 查询向量，形状为 (N, T_q, D_q)
        - key: 键向量，形状为 (N, T_k, D_k)
        - value: 值向量，形状为 (N, T_v, D_v)

        输出:
        - output: 注意力输出，形状为 (N, T_q, D_v)
        """
        # 计算注意力得分
        scores = np.matmul(query, key.transpose(0, 2, 1)) / np.sqrt(self.d_k)
        
        # 计算注意力权重
        weights = self.softmax(scores, axis=-1)
        
        # 计算注意力输出
        output = np.matmul(weights, value)
        
        self.cache = (query, key, value, scores, weights)
        
        return output

    def backward(self, d_output):
        """
        反向传播

        输入:
        - d_output: 上游梯度，形状为 (N, T_q, D_v)

        输出:
        - d_query: 查询向量的梯度，形状为 (N, T_q, D_q)
        - d_key: 键向量的梯度，形状为 (N, T_k, D_k)
        - d_value: 值向量的梯度，形状为 (N, T_v, D_v)
        """
        query, key, value, scores, weights = self.cache
        
        # 计算 d_weights
        d_weights = np.matmul(d_output, value.transpose(0, 2, 1))
        
        # 计算 d_scores
        d_scores = d_weights * weights * (1 - weights)
        
        # 计算 d_query
        d_query = np.matmul(d_scores, key)
        
        # 计算 d_key
        d_key = np.matmul(d_scores.transpose(0, 2, 1), query)
        
        # 计算 d_value
        d_value = np.matmul(weights.transpose(0, 2, 1), d_output)
        
        return d_query, d_key, d_value

# 示例使用
N, T_q, T_k, T_v, D_q, D_k, D_v = 2, 3, 3, 3, 4, 4, 4
query = np.random.randn(N, T_q, D_q)
key = np.random.randn(N, T_k, D_k)
value = np.random.randn(N, T_v, D_v)

attention_layer = AttentionLayer(d_k=D_k, d_v=D_v)
output = attention_layer.forward(query, key, value)

print("Query shape:", query.shape)
print("Key shape:", key.shape)
print("Value shape:", value.shape)
print("Output shape:", output.shape)

# 假设上游梯度为随机值
d_output = np.random.randn(N, T_q, D_v)
d_query, d_key, d_value = attention_layer.backward(d_output)

print("d_query shape:", d_query.shape)
print("d_key shape:", d_key.shape)
print("d_value shape:", d_value.shape)
