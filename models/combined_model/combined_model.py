import torch
from torch import nn


class CombinedModel(nn.Module):
    def __init__(self,
                 acoustic_model: nn.Module,
                 lexical_model: nn.Module,
                 ):
        super().__init__()

        self.acoustic_model = acoustic_model
        self.lexical_model = lexical_model

        self.fc = nn.Linear(
            self.acoustic_model.output_dimension +
            self.lexical_model.output_dimension, 4)

    def forward(self, lexical_input, acoustic_input):

        acoustic_x = self.acoustic_model(acoustic_input.float())
        lexical_x = self.lexical_model(input_ids=lexical_input[0], attention_mask=lexical_input[1])

        # print("acoustic_x", acoustic_x.size())
        # print("lexical_x", lexical_x.size())

        # concat
        concat_x = torch.cat([acoustic_x, lexical_x.view(-1, 128*100)], dim=1)

        return self.fc(concat_x)
