import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoder(nn.Module):

    def __init__(self, n_heads, mlp_dim, input_dim, att_dim):
        super().__init__()
        self.mha = MultiHeadedAttention(n_heads, mlp_dim, input_dim, att_dim)
        self.linear = nn.Linear()

    def forward(self, x):

        out = self.mha(x)
        out = F.layer_norm(x + out)




class MultiHeadedAttention(nn.Module):

    def __init__(self, n_heads, mlp_dim, input_dim, att_dim):
        super().__init__()
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.att_dim = att_dim
        self.attn_heads = nn.ModuleList([ScaledDotProductAttention(input_dim, att_dim) for i in range(n_heads)])

        self.mlp1 = nn.Linear(in_features=input_dim*n_heads, out_features=mlp_dim)
        self.mlp2 = nn.Linear(mlp_dim, input_dim)

    def forward(self, x):
        
        out_all = []
        
        # Get output from each attention head
        for i, l in enumerate(self.attn_heads):
            out = l(x)
            out_all.append(out)

        out_all = torch.cat(out_all, dim=-1)

        linear = self.mlp1(out_all)
        linear = F.relu(linear)
        linear = self.mlp2(linear)
        
        return linear


class ScaledDotProductAttention(nn.Module):

    def __init__(self, input_dim, att_dim):
        super().__init__()

        self.input_dim = input_dim
        self.att_dim = att_dim

        self.key = nn.Linear(input_dim, att_dim, bias=False)
        self.query = nn.Linear(input_dim, att_dim, bias=False)
        self.value = nn.Linear(input_dim, input_dim, bias=False)

    def forward(self, x):

        """
        :param x: tensor (batch_size, sequence_length, embedding_dim)
        :return:
        """

        # Construct keys, queries and values from input
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        # Compute attention
        att = torch.matmul(keys, queries.permute(0, 2, 1)) / torch.sqrt(torch.tensor(self.att_dim))
        att = F.softmax(att, dim=1)

        return torch.matmul(att, values)




def main():

    inputs = torch.rand(size=(4, 32, 128))

    model2 = MultiHeadedAttention(8, 20000, 128, 64)
    out = model2.forward(inputs)
    print(inputs.shape, out.shape)







if __name__ == '__main__':
    main()
