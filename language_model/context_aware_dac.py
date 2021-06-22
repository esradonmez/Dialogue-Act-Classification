import torch
from torch import nn

from .context_aware_attention import ContextAwareAttention
from .utterance_rnn import UtteranceRNN


class ContextAwareDAC(nn.Module):

    def __init__(self, device, model_name="roberta-base", hidden_size=768,
                 output_size=100):
        super(ContextAwareDAC, self).__init__()

        self.output_dimension = 128*100

        self.in_features = 2 * hidden_size

        self.device = device

        # utterance encoder model
        self.utterance_rnn = UtteranceRNN(model_name=model_name,
                                          hidden_size=hidden_size)

        # context aware self attention module
        self.context_aware_attention = ContextAwareAttention(
            hidden_size=2 * hidden_size, output_size=hidden_size,
            seq_len=128)

        # conversaton level rnn
        # self.conversation_rnn = ConversationRNN(input_size=1, hidden_size=hidden_size)

        # classifier on top of feature extractor
        self.classifier = nn.Sequential(*[
            nn.Linear(in_features=self.in_features, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=output_size)
        ])

        # initial hidden_states
        self.hx = torch.randn((2, 1, hidden_size), device=self.device)

        # final project layer, output_features should be changed based on features of
        # cnn model
        self.fc = nn.Linear(in_features=4096, out_features=100, bias=False)

    def forward(self, input_ids, attention_mask):
        """
            batch [input_id, attention_mask, label, audio_feat]

        """
        batch_size, _ = input_ids.shape
        #batch_size = batch[0].shape[0]
        outputs = self.utterance_rnn(
            input_ids=input_ids,
            attention_mask=attention_mask,
            seq_len=[input_ids.shape[1] for _ in range(batch_size)])

        # seq_len=batch['seq_len'].tolist())

        # output size [batch, max_len, hidden_size]

        # create an empty feature vector
        # features = torch.empty((0, self.in_features), device=self.device)

        # hidden
        hx = self.hx

        m = self.context_aware_attention(hidden_states=outputs, h_forward=hx[0].detach())
        #m = torch.flatten(m)
        #final = self.fc(m)

        # for i, x in enumerate(outputs):
        #     x = x.unsqueeze(0)  # x.shape = [1, seq_len, hidden_size]
        #     print('x shape:', x.shape)
        #     # get sentence representation as 2d-matrix and project it linearly
        #     m = self.context_aware_attention(hidden_states=x, h_forward=hx[0].detach())
        #     # [1, output_features = 128, 100]
        #
        #     # flatten
        #     m = torch.flatten(m)
        #     print(m.shape)
        #
        #     final = self.fc(m)
        #     print(final.shape)

            # apply rnn on linearly projected vector
            # hx = self.conversation_rnn(input_=m, hx=hx.detach())

            # concat current utterance's last hidden state to the features vector
            # features = torch.cat((features, hx.view(1, -1)), dim=0)

        self.hx = hx.detach()

        # logits = self.classifier(features)

        return m

    # def forward(self, batch):
    #     """
    #         batch [input_id, attention_mask, label, audio_feat]
    #
    #     """
    #     batch_size = batch[0].shape[0]
    #     seq = [batch[0].shape[1] for i in range(batch_size)]
    #     outputs = self.utterance_rnn(input_ids=batch[0], attention_mask=batch[1],
    #                                  seq_len=seq)
    #     # seq_len=batch['seq_len'].tolist())
    #
    #     # output size [batch, max_len, hidden_size]
    #
    #     # create an empty feature vector
    #     # features = torch.empty((0, self.in_features), device=self.device)
    #
    #     # hidden
    #     hx = self.hx
    #
    #     for i, x in enumerate(outputs):
    #         x = x.unsqueeze(0)  # x.shape = [1, seq_len, hidden_size]
    #         print('x shape:', x.shape)
    #         # get sentence representation as 2d-matrix and project it linearly
    #         m = self.context_aware_attention(hidden_states=x, h_forward=hx[0].detach())
    #         # [1, output_features = 128, 100]
    #
    #         # flatten
    #         m = torch.flatten(m)
    #         print(m.shape)
    #
    #         final = self.fc(m)
    #         print(final.shape)
    #
    #         # apply rnn on linearly projected vector
    #         # hx = self.conversation_rnn(input_=m, hx=hx.detach())
    #
    #         # concat current utterance's last hidden state to the features vector
    #         # features = torch.cat((features, hx.view(1, -1)), dim=0)
    #
    #     self.hx = hx.detach()
    #
    #     # logits = self.classifier(features)
    #
    #     return m
