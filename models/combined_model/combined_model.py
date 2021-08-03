import torch
from torch import nn


class CombinedModel(nn.Module):
    def __init__(self,
                 acoustic_model: nn.Module,
                 lexical_model: nn.Module,
                 attention=False,
                 ):
        super().__init__()

        self.acoustic_model = acoustic_model
        self.lexical_model = lexical_model
        self.attention = attention

        self.attention = nn.Linear(64,1)
        self.pj1 = nn.Linear(128,64)
        self.pj2 = nn.Linear(100*128,64)
        self.pj3 = nn.Linear(128,4)

        self.fc = nn.Linear(
            self.acoustic_model.output_dimension +
            self.lexical_model.output_dimension, 4)

    def forward(self, lexical_input, acoustic_input):

        acoustic_x = self.acoustic_model(acoustic_input.float())
        lexical_x = self.lexical_model(
            input_ids=lexical_input[0],
            attention_mask=lexical_input[1])

        if self.attention:

            acoustic_x = self.pj1(acoustic_x)
            lexical_x = self.pj2(lexical_x)
            acclex = torch.stack([acoustic_x,lexical_x], dim=1)
            # (32,2,64)
            acclex = torch.softmax(self.attention(acclex).squeeze(dim=-1),dim=-1) # (32,2)
            acoustic_weight = acclex[:,0].unsqueeze(dim=1)
            acoustic_weight = acoustic_weight.repeat(1,64)
            lexical_weight = acclex[:, 1].unsqueeze(dim=1)
            lexical_weight = lexical_weight.repeat(1, 64)
            acclex_ = torch.cat([acoustic_weight * acoustic_x, lexical_weight * lexical_x],dim=1)
            # (32,128)

            return self.pj3(acclex_)
        else:
            # concat
            concat_x = torch.cat([acoustic_x, lexical_x.view(-1, 128*100)], dim=1)

        return self.fc(concat_x)
