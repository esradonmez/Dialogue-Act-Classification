from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch.nn as nn
import torch
import time
import datetime
import pandas as pd
from transformers import RobertaTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import numpy as np
import os, json

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


# read in data and creating label dic
def read_data(directory):
    data_train = pd.read_csv(directory + "/train.txt", sep='\t', header=None)
    data_train.columns = ["ind", "label", "text"]
    print(data_train.shape)

    data_dev = pd.read_csv(directory + "/dev.txt", sep='\t', header=None)
    data_dev.columns = ["ind", "label", "text"]
    print(data_dev.shape)

    data_test = pd.read_csv(directory + "/test.txt", sep='\t', header=None)
    data_test.columns = ["ind", "label", "text"]
    print(data_dev.shape)

    label_set = data_train.label.values
    distinct_labels = list(set(label_set))
    label_dic = {distinct_labels[i]: i for i in range(len(distinct_labels))}
    print(label_dic)

    return data_train, data_dev, data_test, label_dic


# encoding texts, max_len can be tuned based on input data
def encoding_text(sentences, label, label_dic):
    # encode labels
    label_encoded = [label_dic[l] for l in label]
    label_encoded = torch.tensor(label_encoded)

    # encode texts
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=128,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    # Print sentence 0, now as a list of IDs.
    print('Original: ', sentences[0])
    print('Token IDs:', input_ids[0])

    return input_ids, attention_masks, label_encoded


# align text and mfcc audio features
def align_text_audio(text_input, audio_input):
    # audio_input {index: mfcc_feature}
    # text_input DataFrame (ind,label,text)

    ind = text_input.ind.values
    audio = [audio_input[idx + '.wav'] for idx in ind]

    return audio


# padding audio array to a certain fix size

def padding(list_):  # list of array
    padded = []
    max_len = 0
    for array in list_:
        arr = np.array(array)
        max_len = max(max_len, arr.shape[0])
    print(max_len)  # maxsize 3360

    for array in list_:
        arr = np.array(array)
        if arr.shape[0] < 1000:  # chopping to 1000 features
            pad = 1000 - arr.shape[0]
            new = np.zeros((pad, 13))
            padded.append(np.concatenate((arr, new), axis=0))
        else:
            padded.append(arr[:1000])
    return padded


# dataloader for both lexical and acoustic models
def creating_dataloader(pd, audio_feat, label_dic):  # input: dataframe, mfcc_features, label_dic
    audio = align_text_audio(pd, audio_feat)
    audio = padding(audio)
    audio = torch.tensor(audio)
    text = pd.text.values
    label = pd.label.values
    input_ids, attention_masks, label_encoded = encoding_text(text, label, label_dic)
    dataset = TensorDataset(input_ids, attention_masks, label_encoded, audio)

    # The DataLoader needs to know our batch size for training, so we specify it
    # here. For fine-tuning BERT on a specific task, the authors recommend a batch
    # size of 16 or 32.
    batch_size = 32

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    dataloader = DataLoader(
        dataset,  # The training samples.
        sampler=RandomSampler(dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )

    return dataloader


class UtteranceRNN(nn.Module):

    def __init__(self, model_name="roberta-base", hidden_size=768, bidirectional=True, num_layers=1):
        super(UtteranceRNN, self).__init__()

        # embedding layer is replaced by pretrained roberta's embedding
        self.base = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name)
        # freeze the model parameters
        for param in self.base.parameters():
            param.requires_grad = False

        # self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        self.rnn = nn.RNN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )

    def forward(self, input_ids, attention_mask, seq_len):
        """
            x.shape = [batch_size, seq_len]
        """

        hidden_states, _ = self.base(input_ids, attention_mask)  # hidden_states.shape = [batch, max_len, hidden_size]

        # padding and packing
        # packed_hidden_states = nn.utils.rnn.pack_padded_sequence(hidden_states, seq_len, batch_first=True, enforce_sorted=False)

        # packed_outputs, _ = self.rnn(packed_hidden_states)

        # packed_outputs is a packed sequence containing all hidden states
        # hidden is now from the final non-padded element in the batch

        # outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

        outputs, _ = self.rnn(hidden_states)

        print("Utterance shape", outputs.shape)  # [batch, max_len, hidden_size=1536]

        return outputs


class ContextAwareAttention(nn.Module):

    def __init__(self, hidden_size=1536, output_size=768, seq_len=128):
        super(ContextAwareAttention, self).__init__()

        # context aware self attention
        self.fc_1 = nn.Linear(in_features=hidden_size, out_features=output_size, bias=False)
        self.fc_3 = nn.Linear(in_features=hidden_size // 2, out_features=output_size, bias=True)
        self.fc_2 = nn.Linear(in_features=output_size, out_features=128, bias=False)

        # linear projection
        self.linear_projection = nn.Linear(in_features=hidden_size, out_features=100, bias=True)

    def forward(self, hidden_states, h_forward):
        """
            hidden_states.shape = [batch, seq_len, hidden_size]
            h_forward.shape = [1, hidden_size]
        """

        # compute the energy
        S = self.fc_2(torch.tanh(self.fc_1(hidden_states) + self.fc_3(h_forward.unsqueeze(1))))
        # S.shape = [1, max_len, output_features] # input_size is hyperparameter
        # print(S.shape)
        # compute the attention
        A = S.softmax(dim=-1)  # S.shape = [1, max_len, output_features]
        # print(A.shape)
        # Compute the sentence representation
        M = torch.matmul(A.permute(0, 2, 1), hidden_states)  # M.shape = [1, output_features, hidden_states]
        # print(M.shape)

        # linear projection of the sentence
        x = self.linear_projection(M)
        print("self attention outputsize", x.shape)  # [1, output_features, 100]
        return x


class ContextAwareDAC(nn.Module):

    def __init__(self, device, model_name="roberta-base", hidden_size=768, output_size=100):
        super(ContextAwareDAC, self).__init__()

        self.in_features = 2 * hidden_size

        self.device = device

        # utterance encoder model
        self.utterance_rnn = UtteranceRNN(model_name=model_name, hidden_size=hidden_size)

        # context aware self attention module
        self.context_aware_attention = ContextAwareAttention(hidden_size=2 * hidden_size, output_size=hidden_size,
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

        # final project layer, output_features should be changed based on features of cnn model
        self.fc = nn.Linear(in_features=12800, out_features=100, bias=False)

    def forward(self, batch):
        """
            batch [input_id, attention_mask, label, audio_feat]

        """
        batch_size = batch[0].shape[0]
        seq = [batch[0].shape[1] for i in range(batch_size)]
        outputs = self.utterance_rnn(input_ids=batch[0], attention_mask=batch[1], seq_len=seq)
        # seq_len=batch['seq_len'].tolist())

        # output size [batch, max_len, hidden_size]

        # create an empty feature vector
        # features = torch.empty((0, self.in_features), device=self.device)

        # hidden
        hx = self.hx

        for i, x in enumerate(outputs):
            x = x.unsqueeze(0)  # x.shape = [1, seq_len, hidden_size]
            print('x shape:', x.shape)
            # get sentence representation as 2d-matrix and project it linearly
            m = self.context_aware_attention(hidden_states=x, h_forward=hx[0].detach())
            # [1, output_features = 128, 100]

            # flatten
            m = torch.flatten(m)
            print(m.shape)

            final = self.fc(m)
            print(final.shape)

            # apply rnn on linearly projected vector
            # hx = self.conversation_rnn(input_=m, hx=hx.detach())

            # concat current utterance's last hidden state to the features vector
            # features = torch.cat((features, hx.view(1, -1)), dim=0)

        self.hx = hx.detach()

        # logits = self.classifier(features)

        return m



def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def main():
    # read in data from text and mfcc files
    directory = os.getcwd()
    data_train, data_dev, data_test, label_dic = read_data(directory)
    with open ('audio_feat.json', 'r') as f:
        audio_feat = json.load(f)
    print("end loading audio features")

    train_dataloader = creating_dataloader(data_train, audio_feat, label_dic)
    dev_dataloader = creating_dataloader(data_dev, audio_feat, label_dic)
    # test_dataloader = creating_dataloader(data_test, audio_feat, label_dic)

    # batch size
    # input id torch.Size([32, 128])
    # attention mask torch.Size([32, 128])
    # label torch.Size([32])
    # audio torch.Size([32, 1000, 13])

    model = ContextAwareDAC(device='cpu')

if __name__ == "__main__":
    main()
