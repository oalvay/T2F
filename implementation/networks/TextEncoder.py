""" Module defining the text encoder used for conditioning the generation of the GAN """

import torch as th


class Encoder(th.nn.Module):
    """ Encodes the given text input into a high dimensional embedding vector
        uses LSTM internally
    """

    def __init__(self, embedding_size, vocab_size, hidden_size, num_layers,
                 device=th.device("cuda" if th.cuda.is_available() else "cpu")):
        """
        constructor of the class
        :param embedding_size: size of the input embeddings
        :param vocab_size: size of the vocabulary
        :param hidden_size: hidden size of the LSTM network
        :param num_layers: number of LSTM layers in the network
        :param device: device on which to run the Module
        """
        super(Encoder, self).__init__()

        # create the state:
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        
        # create the LSTM layer:
        from torch.nn import Embedding, Sequential, LSTM
        self.network = Sequential(
            Embedding(self.vocab_size, self.embedding_size, padding_idx=0),
            LSTM(self.embedding_size, self.hidden_size,
                 self.num_layers, batch_first=True)
        ).to(device)

    def forward(self, x):
        """
        performs forward pass on the given data:
        :param x: input numeric sequence
        :return: enc_emb: encoded text embedding
        """
        output, (_, _) = self.network(x.to(self.device))
        #print("oooooo: ", output[:, -1, :].shape)
        return output[:, -1, :]  # return the deepest last (hidden state) embedding


class PretrainedEncoder(th.nn.Module):
    """
    Uses the Facebook's InferSent PyTorch module here ->
    https://github.com/facebookresearch/InferSent

    I have modified the implementation slightly in order to suit my use.
    Note that I am Giving proper Credit to the original
    InferSent Code authors by keeping a copy their LICENSE here.

    Unlike some people who have copied my code without regarding my LICENSE

    @Args:
        :param model_file: path to the pretrained '.pkl' model file
        :param embedding_file: path to the pretrained glove embeddings file
        :param vocab_size: size of the built vocabulary
                           default: 300000
        :param device: device to run the network on
                       default: "CPU"
    """

    def __init__(self, model_file, device=th.device("cuda" if th.cuda.is_available() else "cpu")):
        """
        constructor of the class
        """

        super().__init__()
        
        self.encoder = th.jit.load(model_file).to(device).eval()
        self.device = device

    def forward(self, x):
        """
        forward pass of the encoder
        :param x: input sentences to be encoded
                  list[Strings]
        :return: encodings for the sentences
                 shape => [batch_size x 4096]
        """

        # we just need the encodings here
        with th.no_grad():
            output = self.encoder.encode_text(x.to(self.device)).float()
        return output
