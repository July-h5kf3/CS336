import torch
from einops import rearrange, einsum

class Embedding(torch.nn.Module):
    def __init__(self,num_embeddings,embedding_dim,device = None,dtype = None):
        """
        num_embeddings: int 表示词表大小
        embedding_dim: int 表示每个词向量的维度
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.nn.Parameter(torch.empty((num_embeddings,embedding_dim),device = device,dtype = dtype))
        torch.nn.init.trunc_normal_(self.weight,mean = 0.0,std = 1.0,a = -3.0,b = 3.0)
    def forward(self,token_ids):
        """
        根据给定的token_ids返回对应的Embedding向量
        """
        return self.weight[token_ids]