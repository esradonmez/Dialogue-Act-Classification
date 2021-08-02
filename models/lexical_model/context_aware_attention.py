import torch
from torch import nn


class ContextAwareAttention(nn.Module):

    def __init__(self, hidden_size=1536, output_size=768, seq_len=128):
        super().__init__()

        # context aware self attention
        self.fc_1 = nn.Linear(in_features=hidden_size, out_features=output_size,
                              bias=False)
        self.fc_3 = nn.Linear(in_features=hidden_size // 2, out_features=output_size,
                              bias=True)
        self.fc_2 = nn.Linear(in_features=output_size, out_features=128, bias=False)

        # linear projection
        self.linear_projection = nn.Linear(in_features=hidden_size , out_features=100,
                                           bias=True)

    def forward(self, hidden_states, h_forward):
        # compute the energy
        S = self.fc_2(
            torch.tanh(self.fc_1(hidden_states) + self.fc_3(h_forward.unsqueeze(1))))

        # compute the attention
        A = S.softmax(dim=-1)

        # Compute the sentence representation
        M = torch.matmul(A.permute(0, 2, 1), hidden_states)  # (batch,max_len,1536)
        batch_size= M.shape[0]
        # linear projection of the sentence
        x = self.linear_projection(M)
        x = x.view(batch_size,-1,100)

        return x
