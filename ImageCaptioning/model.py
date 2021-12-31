import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(num_embeddings=self.vocab_size,
                                  embedding_dim=self.embed_size)
        self.lstm = nn.LSTM(input_size=self.embed_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(in_features=self.hidden_size,
                            out_features=self.vocab_size)



    def forward(self, features, captions):

        captions_embed = self.embed(captions[:, :-1])
        captions_embed = torch.cat((features.unsqueeze(1), captions_embed), 1)
        
        out, _hc = self.lstm(captions_embed)
        
        outputs = self.fc(out)

        return outputs


    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        ids = []
        for i in range(max_len):
            out, states = self.lstm(inputs, states)
            outputs = self.fc(out.squeeze(1))
            predicted = outputs.max(1)[1]
            ids.append(predicted.tolist()[0])
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)

        return ids
