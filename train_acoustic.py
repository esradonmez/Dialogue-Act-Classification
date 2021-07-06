from dataset import LABEL_MAP
from models import SpeechCnn
from utils import train

if __name__ == '__main__':
    # define model parameters
    conv_kernel = (10, 13)
    pool_kernel = (10, 1)
    pool_stride = 6
    model = SpeechCnn(conv_kernel=conv_kernel, pool_kernel=pool_kernel,
                      pool_stride=pool_stride, output_dimension=len(LABEL_MAP))

    # define training parameters
    learning_rate = 0.0001
    batch_size = 32
    epochs = 50

    train.train(model, model_type="acoustic",
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=epochs)
