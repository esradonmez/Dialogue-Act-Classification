import torch
from torch import nn


class CombinedModel(nn.Module):
    def __init__(self,
                 acoustic_model: nn.Module,
                 lexical_model: nn.Module,
                 lexical_tokenizer
                 ):
        super().__init__()

        self.acoustic_model = acoustic_model
        self.lexical_model = lexical_model
        self.lexical_tokenizer = lexical_tokenizer

        self.fc = nn.Linear(
            self.acoustic_model.output_dimension +
            self.lexical_model.output_dimension, 4)

    def forward(self, combined_input):
        lexical_input, acoustic_input = combined_input

        acoustic_x = self.acoustic_model(acoustic_input.float())
        lexical_x = self.lexical_model(**self.lexical_tokenizer(
            lexical_input,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=128,  # Pad & truncate all sentences.
            padding='max_length',
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        ))

        # print("acoustic_x", acoustic_x.size())
        # print("lexical_x", lexical_x.size())

        # concat
        concat_x = torch.cat([acoustic_x, lexical_x.view(-1, 128*100)], dim=1)

        return self.fc(concat_x)
