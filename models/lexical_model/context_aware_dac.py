import os
import torch
from dotenv import load_dotenv
from torch import nn

from .context_aware_attention import ContextAwareAttention
from .utterance_rnn import UtteranceRNN

load_dotenv(".env")
DEVICE = torch.device(
    os.getenv("SDS_DAC_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))


class ContextAwareDAC(nn.Module):

    def __init__(self, mode="all", model_name="roberta-base", hidden_size=768,
                 output_size=100):
        super().__init__()
        self.mode = mode
        self.output_dimension = 128 * 100
        self.in_features = 2 * hidden_size

        # utterance encoder model
        self.utterance_rnn = UtteranceRNN(model_name=model_name,
                                          hidden_size=hidden_size,
                                          mode=mode)

        # context aware self attention module
        self.context_aware_attention = ContextAwareAttention(
            hidden_size=2 * hidden_size, output_size=hidden_size,
            seq_len=128)

        # classifier on top of feature extractor
        self.classifier = nn.Sequential(*[
            nn.Linear(in_features=self.in_features, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=output_size)
        ])

        # initial hidden_states
        self.hx = torch.randn((2, 1, hidden_size), device=DEVICE)

        # final project layer, output_features should be changed based on features of
        # cnn model

        self.fc1 = nn.Linear(in_features=768, out_features=100, bias=False)
        self.fc2 = nn.Linear(in_features=1536, out_features=100, bias=False)

    def forward(self, input_ids, attention_mask):
        """
            batch [input_id, attention_mask, label, audio_feat]
        """

        batch_size, _ = input_ids.shape
        outputs = self.utterance_rnn(
            input_ids=input_ids,
            attention_mask=attention_mask,
            seq_len=[input_ids.shape[1] for _ in range(batch_size)])
        if self.mode == "onlylm":
            return self.fc1(outputs)  # (batch_size,max_len,100)

        elif self.mode == "lmrnn":
            return self.fc2(outputs)  # (batch_size,max_len,100)

        else:
            # hidden
            hx = self.hx
            m = self.context_aware_attention(
                hidden_states=outputs,
                h_forward=hx[0].detach())
            self.hx = hx.detach()
            m = m.view(batch_size,-1)
            return m
