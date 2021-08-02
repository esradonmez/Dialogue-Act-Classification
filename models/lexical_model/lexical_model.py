
from torch import nn
from.context_aware_dac import ContextAwareDAC

class LexicalModel(nn.Module):
    def __init__(self,
                 mode="all",
                 output_dimension: int = 4
                 ):
        super().__init__()

        self.output_dimension = output_dimension

        self.dac = ContextAwareDAC(mode=mode)

        self.fc = nn.Linear(
            128*100,
            self.output_dimension)

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        output = self.dac(input_ids, attention_mask)
        output = output.view(batch_size,-1)

        return self.fc(output)
