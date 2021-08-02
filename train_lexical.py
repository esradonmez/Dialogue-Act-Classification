# Used only for training lexical model
# Three modes:
#   default "all" includes pretrained roberta, rnn and attention layers
#           "onlylm" includes only pretrained roberta
#           "lmrnn" includes pretrained roberta and rnn layers 

from models import LexicalModel
from utils import train

def run(mode):

    model = LexicalModel(mode)

    # define training parameters
    learning_rate = 0.0001
    batch_size = 32
    epochs = 50

    train.train(model, model_type="lexical",
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=epochs)


if __name__ == '__main__':
   run(mode="lmrnn")
