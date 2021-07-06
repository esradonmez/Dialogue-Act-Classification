import torch
from torch import nn


class ContextAwareAttention(nn.Module):

    def __init__(self, hidden_size=1536, output_size=768, seq_len=128):
        super(ContextAwareAttention, self).__init__()

        # context aware self attention
        self.fc_1 = nn.Linear(in_features=hidden_size, out_features=output_size,
                              bias=False)
        self.fc_3 = nn.Linear(in_features=hidden_size // 2, out_features=output_size,
                              bias=True)
        self.fc_2 = nn.Linear(in_features=output_size, out_features=128, bias=False)

        # linear projection
        self.linear_projection = nn.Linear(in_features=hidden_size, out_features=100,
                                           bias=True)

    def forward(self, hidden_states, h_forward):
        """
            hidden_states.shape = [batch, seq_len, hidden_size]
            h_forward.shape = [1, hidden_size]
        """

        # compute the energy
        S = self.fc_2(
            torch.tanh(self.fc_1(hidden_states) + self.fc_3(h_forward.unsqueeze(1))))
        # S.shape = [1, max_len, output_features] # input_size is hyperparameter
        # print(S.shape)
        # compute the attention
        A = S.softmax(dim=-1)  # S.shape = [1, max_len, output_features]
        # print(A.shape)
        # Compute the sentence representation
        M = torch.matmul(A.permute(0, 2, 1),
                         hidden_states)  # M.shape = [1, output_features, hidden_states]
        # print(M.shape)

        # linear projection of the sentence
        x = self.linear_projection(M)
        #print("self attention outputsize", x.shape)  # [1, output_features, 100]
        return x
