from models import CombinedModel, ContextAwareDAC, SpeechCnn
from utils import train

if __name__ == '__main__':
    # define acoustic_model parameters
    conv_kernel = (10, 13)
    pool_kernel = (10, 1)
    pool_stride = 6
    acoustic_model = SpeechCnn(conv_kernel=conv_kernel, pool_kernel=pool_kernel,
                               pool_stride=pool_stride, output_dimension=128)

    # define lexical_model parameters
    lexical_model = ContextAwareDAC()

    # define combined model parameters
    model = CombinedModel(
        acoustic_model=acoustic_model,
        lexical_model=lexical_model,
        attention=True
    )

    # define training parameters
    learning_rate = 0.0001
    batch_size = 32
    epochs = 50

    train.train(model, model_type="combined",
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=epochs)
