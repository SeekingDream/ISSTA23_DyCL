import torch
import torch.nn as nn
from collections import namedtuple
import torch.nn.functional as F
import numpy as np

class MyDNN(nn.Module):

    def __init__(self, dnn):
        super(MyDNN, self).__init__()
        self.dnn = dnn
        base_name = dir(nn.Linear(10, 10))
        for p in dir(dnn):
            if p not in base_name:
                setattr(self, p, eval('dnn.%s' % p))
                print(p)

    def forward(self, x):
        return self.model(x)

    def my_func0(self, input_dict):
        x = input_dict['x']
        encoder_out = self.encoder(x)
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(3)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        k_prev_words = torch.LongTensor([[self.word_map['<start>']]] * batch_size).to(x.device)
        seqs = k_prev_words
        (h, c) = self.decoder.init_hidden_state(encoder_out)
        embeddings = self.decoder.embedding(k_prev_words).squeeze(1)
        (awe, alpha) = self.decoder.attention(encoder_out, h)
        gate = self.decoder.sigmoid(self.decoder.f_beta(h))
        awe = gate * awe
        (h, c) = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
        scores = self.decoder.fc(h)
        (_, next_word_inds) = scores.max(1)
        seqs = torch.cat([seqs, next_word_inds.unsqueeze(1)], dim=1)
        return (seqs, encoder_out, k_prev_words, h, c, next_word_inds)

    def my_func0_onnx(self, x):
        encoder_out = self.encoder(x)
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(3)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        k_prev_words = torch.LongTensor([[self.word_map['<start>']]] * batch_size).to(x.device)
        seqs = k_prev_words
        (h, c) = self.decoder.init_hidden_state(encoder_out)
        embeddings = self.decoder.embedding(k_prev_words).squeeze(1)
        (awe, alpha) = self.decoder.attention(encoder_out, h)
        gate = self.decoder.sigmoid(self.decoder.f_beta(h))
        awe = gate * awe
        (h, c) = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
        scores = self.decoder.fc(h)
        (_, next_word_inds) = scores.max(1)
        seqs = torch.cat([seqs, next_word_inds.unsqueeze(1)], dim=1)
        return (seqs, encoder_out, k_prev_words, h, c, next_word_inds)

    def my_func1(self, input_dict):
        seqs = input_dict['seqs']
        return seqs

    def my_func1_onnx(self, seqs):
        return seqs

    def my_func2(self, input_dict):
        (encoder_out, k_prev_words, seqs, h, c) = (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c'])
        embeddings = self.decoder.embedding(k_prev_words).squeeze(1)
        (awe, alpha) = self.decoder.attention(encoder_out, h)
        gate = self.decoder.sigmoid(self.decoder.f_beta(h))
        awe = gate * awe
        (h, c) = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
        scores = self.decoder.fc(h)
        (_, next_word_inds) = scores.max(1)
        seqs = torch.cat([seqs, next_word_inds.unsqueeze(1)], dim=1)
        return (seqs, encoder_out, k_prev_words, h, c, next_word_inds)

    def my_func2_onnx(self, encoder_out, k_prev_words, seqs, h, c):
        embeddings = self.decoder.embedding(k_prev_words).squeeze(1)
        (awe, alpha) = self.decoder.attention(encoder_out, h)
        gate = self.decoder.sigmoid(self.decoder.f_beta(h))
        awe = gate * awe
        (h, c) = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
        scores = self.decoder.fc(h)
        (_, next_word_inds) = scores.max(1)
        seqs = torch.cat([seqs, next_word_inds.unsqueeze(1)], dim=1)
        return (seqs, encoder_out, k_prev_words, h, c, next_word_inds)

    def my_func3(self, input_dict):
        seqs = input_dict['seqs']
        return seqs

    def my_func3_onnx(self, seqs):
        return seqs

    def my_func4(self, input_dict):
        (encoder_out, k_prev_words, seqs, h, c) = (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c'])
        embeddings = self.decoder.embedding(k_prev_words).squeeze(1)
        (awe, alpha) = self.decoder.attention(encoder_out, h)
        gate = self.decoder.sigmoid(self.decoder.f_beta(h))
        awe = gate * awe
        (h, c) = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
        scores = self.decoder.fc(h)
        (_, next_word_inds) = scores.max(1)
        seqs = torch.cat([seqs, next_word_inds.unsqueeze(1)], dim=1)
        return (seqs, encoder_out, k_prev_words, h, c, next_word_inds)

    def my_func4_onnx(self, encoder_out, k_prev_words, seqs, h, c):
        embeddings = self.decoder.embedding(k_prev_words).squeeze(1)
        (awe, alpha) = self.decoder.attention(encoder_out, h)
        gate = self.decoder.sigmoid(self.decoder.f_beta(h))
        awe = gate * awe
        (h, c) = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
        scores = self.decoder.fc(h)
        (_, next_word_inds) = scores.max(1)
        seqs = torch.cat([seqs, next_word_inds.unsqueeze(1)], dim=1)
        return (seqs, encoder_out, k_prev_words, h, c, next_word_inds)

    def my_func5(self, input_dict):
        seqs = input_dict['seqs']
        return seqs

    def my_func5_onnx(self, seqs):
        return seqs

    def my_func6(self, input_dict):
        (encoder_out, k_prev_words, seqs, h, c) = (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c'])
        embeddings = self.decoder.embedding(k_prev_words).squeeze(1)
        (awe, alpha) = self.decoder.attention(encoder_out, h)
        gate = self.decoder.sigmoid(self.decoder.f_beta(h))
        awe = gate * awe
        (h, c) = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
        scores = self.decoder.fc(h)
        (_, next_word_inds) = scores.max(1)
        seqs = torch.cat([seqs, next_word_inds.unsqueeze(1)], dim=1)
        return (seqs, encoder_out, k_prev_words, h, c, next_word_inds)

    def my_func6_onnx(self, encoder_out, k_prev_words, seqs, h, c):
        embeddings = self.decoder.embedding(k_prev_words).squeeze(1)
        (awe, alpha) = self.decoder.attention(encoder_out, h)
        gate = self.decoder.sigmoid(self.decoder.f_beta(h))
        awe = gate * awe
        (h, c) = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
        scores = self.decoder.fc(h)
        (_, next_word_inds) = scores.max(1)
        seqs = torch.cat([seqs, next_word_inds.unsqueeze(1)], dim=1)
        return (seqs, encoder_out, k_prev_words, h, c, next_word_inds)

    def my_func7(self, input_dict):
        seqs = input_dict['seqs']
        return seqs

    def my_func7_onnx(self, seqs):
        return seqs

    def my_func8(self, input_dict):
        (encoder_out, k_prev_words, seqs, h, c) = (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c'])
        embeddings = self.decoder.embedding(k_prev_words).squeeze(1)
        (awe, alpha) = self.decoder.attention(encoder_out, h)
        gate = self.decoder.sigmoid(self.decoder.f_beta(h))
        awe = gate * awe
        (h, c) = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
        scores = self.decoder.fc(h)
        (_, next_word_inds) = scores.max(1)
        seqs = torch.cat([seqs, next_word_inds.unsqueeze(1)], dim=1)
        return (seqs, encoder_out, k_prev_words, h, c, next_word_inds)

    def my_func8_onnx(self, encoder_out, k_prev_words, seqs, h, c):
        embeddings = self.decoder.embedding(k_prev_words).squeeze(1)
        (awe, alpha) = self.decoder.attention(encoder_out, h)
        gate = self.decoder.sigmoid(self.decoder.f_beta(h))
        awe = gate * awe
        (h, c) = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
        scores = self.decoder.fc(h)
        (_, next_word_inds) = scores.max(1)
        seqs = torch.cat([seqs, next_word_inds.unsqueeze(1)], dim=1)
        return (seqs, encoder_out, k_prev_words, h, c, next_word_inds)

    def my_func9(self, input_dict):
        seqs = input_dict['seqs']
        return seqs

    def my_func9_onnx(self, seqs):
        return seqs

    def my_func10(self, input_dict):
        (encoder_out, k_prev_words, seqs, h, c) = (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c'])
        embeddings = self.decoder.embedding(k_prev_words).squeeze(1)
        (awe, alpha) = self.decoder.attention(encoder_out, h)
        gate = self.decoder.sigmoid(self.decoder.f_beta(h))
        awe = gate * awe
        (h, c) = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
        scores = self.decoder.fc(h)
        (_, next_word_inds) = scores.max(1)
        seqs = torch.cat([seqs, next_word_inds.unsqueeze(1)], dim=1)
        return (seqs, encoder_out, k_prev_words, h, c, next_word_inds)

    def my_func10_onnx(self, encoder_out, k_prev_words, seqs, h, c):
        embeddings = self.decoder.embedding(k_prev_words).squeeze(1)
        (awe, alpha) = self.decoder.attention(encoder_out, h)
        gate = self.decoder.sigmoid(self.decoder.f_beta(h))
        awe = gate * awe
        (h, c) = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
        scores = self.decoder.fc(h)
        (_, next_word_inds) = scores.max(1)
        seqs = torch.cat([seqs, next_word_inds.unsqueeze(1)], dim=1)
        return (seqs, encoder_out, k_prev_words, h, c, next_word_inds)

    def my_func11(self, input_dict):
        seqs = input_dict['seqs']
        return seqs

    def my_func11_onnx(self, seqs):
        return seqs

    def my_func12(self, input_dict):
        (encoder_out, k_prev_words, seqs, h, c) = (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c'])
        embeddings = self.decoder.embedding(k_prev_words).squeeze(1)
        (awe, alpha) = self.decoder.attention(encoder_out, h)
        gate = self.decoder.sigmoid(self.decoder.f_beta(h))
        awe = gate * awe
        (h, c) = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
        scores = self.decoder.fc(h)
        (_, next_word_inds) = scores.max(1)
        seqs = torch.cat([seqs, next_word_inds.unsqueeze(1)], dim=1)
        return (seqs, encoder_out, k_prev_words, h, c, next_word_inds)

    def my_func12_onnx(self, encoder_out, k_prev_words, seqs, h, c):
        embeddings = self.decoder.embedding(k_prev_words).squeeze(1)
        (awe, alpha) = self.decoder.attention(encoder_out, h)
        gate = self.decoder.sigmoid(self.decoder.f_beta(h))
        awe = gate * awe
        (h, c) = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
        scores = self.decoder.fc(h)
        (_, next_word_inds) = scores.max(1)
        seqs = torch.cat([seqs, next_word_inds.unsqueeze(1)], dim=1)
        return (seqs, encoder_out, k_prev_words, h, c, next_word_inds)

    def my_func13(self, input_dict):
        seqs = input_dict['seqs']
        return seqs

    def my_func13_onnx(self, seqs):
        return seqs

    def my_func14(self, input_dict):
        (encoder_out, k_prev_words, seqs, h, c) = (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c'])
        embeddings = self.decoder.embedding(k_prev_words).squeeze(1)
        (awe, alpha) = self.decoder.attention(encoder_out, h)
        gate = self.decoder.sigmoid(self.decoder.f_beta(h))
        awe = gate * awe
        (h, c) = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
        scores = self.decoder.fc(h)
        (_, next_word_inds) = scores.max(1)
        seqs = torch.cat([seqs, next_word_inds.unsqueeze(1)], dim=1)
        return (seqs, encoder_out, k_prev_words, h, c, next_word_inds)

    def my_func14_onnx(self, encoder_out, k_prev_words, seqs, h, c):
        embeddings = self.decoder.embedding(k_prev_words).squeeze(1)
        (awe, alpha) = self.decoder.attention(encoder_out, h)
        gate = self.decoder.sigmoid(self.decoder.f_beta(h))
        awe = gate * awe
        (h, c) = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
        scores = self.decoder.fc(h)
        (_, next_word_inds) = scores.max(1)
        seqs = torch.cat([seqs, next_word_inds.unsqueeze(1)], dim=1)
        return (seqs, encoder_out, k_prev_words, h, c, next_word_inds)

    def my_func15(self, input_dict):
        seqs = input_dict['seqs']
        return seqs

    def my_func15_onnx(self, seqs):
        return seqs

    def my_func16(self, input_dict):
        (encoder_out, k_prev_words, seqs, h, c) = (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c'])
        embeddings = self.decoder.embedding(k_prev_words).squeeze(1)
        (awe, alpha) = self.decoder.attention(encoder_out, h)
        gate = self.decoder.sigmoid(self.decoder.f_beta(h))
        awe = gate * awe
        (h, c) = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
        scores = self.decoder.fc(h)
        (_, next_word_inds) = scores.max(1)
        seqs = torch.cat([seqs, next_word_inds.unsqueeze(1)], dim=1)
        return (seqs, h, c, encoder_out, k_prev_words, next_word_inds)

    def my_func16_onnx(self, encoder_out, k_prev_words, seqs, h, c):
        embeddings = self.decoder.embedding(k_prev_words).squeeze(1)
        (awe, alpha) = self.decoder.attention(encoder_out, h)
        gate = self.decoder.sigmoid(self.decoder.f_beta(h))
        awe = gate * awe
        (h, c) = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
        scores = self.decoder.fc(h)
        (_, next_word_inds) = scores.max(1)
        seqs = torch.cat([seqs, next_word_inds.unsqueeze(1)], dim=1)
        return (seqs, h, c, encoder_out, k_prev_words, next_word_inds)

    def my_func17(self, input_dict):
        seqs = input_dict['seqs']
        return seqs

    def my_func17_onnx(self, seqs):
        return seqs

    def my_func18(self, input_dict):
        (seqs, h, c, encoder_out, k_prev_words) = (input_dict['seqs'], input_dict['h'], input_dict['c'], input_dict['encoder_out'], input_dict['k_prev_words'])
        embeddings = self.decoder.embedding(k_prev_words).squeeze(1)
        (awe, alpha) = self.decoder.attention(encoder_out, h)
        gate = self.decoder.sigmoid(self.decoder.f_beta(h))
        awe = gate * awe
        (h, c) = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
        scores = self.decoder.fc(h)
        (_, next_word_inds) = scores.max(1)
        seqs = torch.cat([seqs, next_word_inds.unsqueeze(1)], dim=1)
        return (seqs, next_word_inds)

    def my_func18_onnx(self, seqs, h, c, encoder_out, k_prev_words):
        embeddings = self.decoder.embedding(k_prev_words).squeeze(1)
        (awe, alpha) = self.decoder.attention(encoder_out, h)
        gate = self.decoder.sigmoid(self.decoder.f_beta(h))
        awe = gate * awe
        (h, c) = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
        scores = self.decoder.fc(h)
        (_, next_word_inds) = scores.max(1)
        seqs = torch.cat([seqs, next_word_inds.unsqueeze(1)], dim=1)
        return (seqs, next_word_inds)

    def my_func19(self, input_dict):
        seqs = input_dict['seqs']
        return seqs

    def my_func19_onnx(self, seqs):
        return seqs

    def my_func20(self, input_dict):
        seqs = input_dict['seqs']
        return seqs

    def my_func20_onnx(self, seqs):
        return seqs


def predictAPI_No_OPT(input_dict, model_dict, self, constant_dict):
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func0'](input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict['seqs'] = seqs
        seqs = model_dict['my_func1'](input_dict)
        return seqs
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func2'](input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict['seqs'] = seqs
        seqs = model_dict['my_func3'](input_dict)
        return seqs
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func4'](input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict['seqs'] = seqs
        seqs = model_dict['my_func5'](input_dict)
        return seqs
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func6'](input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict['seqs'] = seqs
        seqs = model_dict['my_func7'](input_dict)
        return seqs
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func8'](input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict['seqs'] = seqs
        seqs = model_dict['my_func9'](input_dict)
        return seqs
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func10'](input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict['seqs'] = seqs
        seqs = model_dict['my_func11'](input_dict)
        return seqs
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func12'](input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict['seqs'] = seqs
        seqs = model_dict['my_func13'](input_dict)
        return seqs
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func14'](input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict['seqs'] = seqs
        seqs = model_dict['my_func15'](input_dict)
        return seqs
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    (seqs, h, c, encoder_out, k_prev_words, next_word_inds) = model_dict['my_func16'](input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict['seqs'] = seqs
        seqs = model_dict['my_func17'](input_dict)
        return seqs
    (input_dict['seqs'], input_dict['h'], input_dict['c'], input_dict['encoder_out'], input_dict['k_prev_words']) = (seqs, h, c, encoder_out, k_prev_words)
    (seqs, next_word_inds) = model_dict['my_func18'](input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict['seqs'] = seqs
        seqs = model_dict['my_func19'](input_dict)
        return seqs
    input_dict['seqs'] = seqs
    seqs = model_dict['my_func20'](input_dict)
    return seqs



def ONNX_API_No_OPT(input_dict, model_dict, self, constant_dict):
    [seqs, encoder_out, k_prev_words, h, c, next_word_inds] = model_dict['my_func0'].run(['output::seqs', 'output::encoder_out', 'output::k_prev_words', 'output::h', 'output::c', 'output::next_word_inds'], input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict = {}
        input_dict['input::seqs'] = seqs
        [seqs] = model_dict['my_func1'].run(['output::seqs'], input_dict)
        return seqs
    input_dict = {}
    (input_dict['input::encoder_out'], input_dict['input::k_prev_words'], input_dict['input::seqs'], input_dict['input::h'], input_dict['input::c']) = (encoder_out, k_prev_words, seqs, h, c)
    [seqs, encoder_out, k_prev_words, h, c, next_word_inds] = model_dict['my_func2'].run(['output::seqs', 'output::encoder_out', 'output::k_prev_words', 'output::h', 'output::c', 'output::next_word_inds'], input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict = {}
        input_dict['input::seqs'] = seqs
        [seqs] = model_dict['my_func3'].run(['output::seqs'], input_dict)
        return seqs
    input_dict = {}
    (input_dict['input::encoder_out'], input_dict['input::k_prev_words'], input_dict['input::seqs'], input_dict['input::h'], input_dict['input::c']) = (encoder_out, k_prev_words, seqs, h, c)
    [seqs, encoder_out, k_prev_words, h, c, next_word_inds] = model_dict['my_func4'].run(['output::seqs', 'output::encoder_out', 'output::k_prev_words', 'output::h', 'output::c', 'output::next_word_inds'], input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict = {}
        input_dict['input::seqs'] = seqs
        [seqs] = model_dict['my_func5'].run(['output::seqs'], input_dict)
        return seqs
    input_dict = {}
    (input_dict['input::encoder_out'], input_dict['input::k_prev_words'], input_dict['input::seqs'], input_dict['input::h'], input_dict['input::c']) = (encoder_out, k_prev_words, seqs, h, c)
    [seqs, encoder_out, k_prev_words, h, c, next_word_inds] = model_dict['my_func6'].run(['output::seqs', 'output::encoder_out', 'output::k_prev_words', 'output::h', 'output::c', 'output::next_word_inds'], input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict = {}
        input_dict['input::seqs'] = seqs
        [seqs] = model_dict['my_func7'].run(['output::seqs'], input_dict)
        return seqs
    input_dict = {}
    (input_dict['input::encoder_out'], input_dict['input::k_prev_words'], input_dict['input::seqs'], input_dict['input::h'], input_dict['input::c']) = (encoder_out, k_prev_words, seqs, h, c)
    [seqs, encoder_out, k_prev_words, h, c, next_word_inds] = model_dict['my_func8'].run(['output::seqs', 'output::encoder_out', 'output::k_prev_words', 'output::h', 'output::c', 'output::next_word_inds'], input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict = {}
        input_dict['input::seqs'] = seqs
        [seqs] = model_dict['my_func9'].run(['output::seqs'], input_dict)
        return seqs
    input_dict = {}
    (input_dict['input::encoder_out'], input_dict['input::k_prev_words'], input_dict['input::seqs'], input_dict['input::h'], input_dict['input::c']) = (encoder_out, k_prev_words, seqs, h, c)
    [seqs, encoder_out, k_prev_words, h, c, next_word_inds] = model_dict['my_func10'].run(['output::seqs', 'output::encoder_out', 'output::k_prev_words', 'output::h', 'output::c', 'output::next_word_inds'], input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict = {}
        input_dict['input::seqs'] = seqs
        [seqs] = model_dict['my_func11'].run(['output::seqs'], input_dict)
        return seqs
    input_dict = {}
    (input_dict['input::encoder_out'], input_dict['input::k_prev_words'], input_dict['input::seqs'], input_dict['input::h'], input_dict['input::c']) = (encoder_out, k_prev_words, seqs, h, c)
    [seqs, encoder_out, k_prev_words, h, c, next_word_inds] = model_dict['my_func12'].run(['output::seqs', 'output::encoder_out', 'output::k_prev_words', 'output::h', 'output::c', 'output::next_word_inds'], input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict = {}
        input_dict['input::seqs'] = seqs
        [seqs] = model_dict['my_func13'].run(['output::seqs'], input_dict)
        return seqs
    input_dict = {}
    (input_dict['input::encoder_out'], input_dict['input::k_prev_words'], input_dict['input::seqs'], input_dict['input::h'], input_dict['input::c']) = (encoder_out, k_prev_words, seqs, h, c)
    [seqs, encoder_out, k_prev_words, h, c, next_word_inds] = model_dict['my_func14'].run(['output::seqs', 'output::encoder_out', 'output::k_prev_words', 'output::h', 'output::c', 'output::next_word_inds'], input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict = {}
        input_dict['input::seqs'] = seqs
        [seqs] = model_dict['my_func15'].run(['output::seqs'], input_dict)
        return seqs
    input_dict = {}
    (input_dict['input::encoder_out'], input_dict['input::k_prev_words'], input_dict['input::seqs'], input_dict['input::h'], input_dict['input::c']) = (encoder_out, k_prev_words, seqs, h, c)
    [seqs, h, c, encoder_out, k_prev_words, next_word_inds] = model_dict['my_func16'].run(['output::seqs', 'output::h', 'output::c', 'output::encoder_out', 'output::k_prev_words', 'output::next_word_inds'], input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict = {}
        input_dict['input::seqs'] = seqs
        [seqs] = model_dict['my_func17'].run(['output::seqs'], input_dict)
        return seqs
    input_dict = {}
    (input_dict['input::seqs'], input_dict['input::h'], input_dict['input::c'], input_dict['input::encoder_out'], input_dict['input::k_prev_words']) = (seqs, h, c, encoder_out, k_prev_words)
    [seqs, next_word_inds] = model_dict['my_func18'].run(['output::seqs', 'output::next_word_inds'], input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict = {}
        input_dict['input::seqs'] = seqs
        [seqs] = model_dict['my_func19'].run(['output::seqs'], input_dict)
        return seqs
    input_dict = {}
    input_dict['input::seqs'] = seqs
    [seqs] = model_dict['my_func20'].run(['output::seqs'], input_dict)
    return seqs



def TVM_API_No_OPT(input_dict, model_dict, params_dict, self, constant_dict):
    params = params_dict['my_func0']
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func0'](**params)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        params = params_dict['my_func1']
        input_dict = {}
        input_dict['seqs'] = seqs
        params.update(input_dict)
        seqs = model_dict['my_func1'](**params)
        return seqs
    params = params_dict['my_func2']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func2'](**params)
    params = params_dict['my_func2']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func2'](**params)
    params = params_dict['my_func2']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func2'](**params)
    params = params_dict['my_func2']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func2'](**params)
    params = params_dict['my_func2']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func2'](**params)
    params = params_dict['my_func2']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func2'](**params)
    params = params_dict['my_func2']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func2'](**params)
    params = params_dict['my_func2']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func2'](**params)
    params = params_dict['my_func2']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func2'](**params)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        params = params_dict['my_func3']
        input_dict = {}
        input_dict['seqs'] = seqs
        params.update(input_dict)
        seqs = model_dict['my_func3'](**params)
        return seqs
    params = params_dict['my_func4']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func4'](**params)
    params = params_dict['my_func4']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func4'](**params)
    params = params_dict['my_func4']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func4'](**params)
    params = params_dict['my_func4']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func4'](**params)
    params = params_dict['my_func4']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func4'](**params)
    params = params_dict['my_func4']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func4'](**params)
    params = params_dict['my_func4']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func4'](**params)
    params = params_dict['my_func4']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func4'](**params)
    params = params_dict['my_func4']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func4'](**params)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        params = params_dict['my_func5']
        input_dict = {}
        input_dict['seqs'] = seqs
        params.update(input_dict)
        seqs = model_dict['my_func5'](**params)
        return seqs
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func6'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func6'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func6'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func6'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func6'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func6'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func6'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func6'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func6'](**params)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        params = params_dict['my_func7']
        input_dict = {}
        input_dict['seqs'] = seqs
        params.update(input_dict)
        seqs = model_dict['my_func7'](**params)
        return seqs
    params = params_dict['my_func8']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func8'](**params)
    params = params_dict['my_func8']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func8'](**params)
    params = params_dict['my_func8']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func8'](**params)
    params = params_dict['my_func8']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func8'](**params)
    params = params_dict['my_func8']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func8'](**params)
    params = params_dict['my_func8']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func8'](**params)
    params = params_dict['my_func8']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func8'](**params)
    params = params_dict['my_func8']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func8'](**params)
    params = params_dict['my_func8']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func8'](**params)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        params = params_dict['my_func9']
        input_dict = {}
        input_dict['seqs'] = seqs
        params.update(input_dict)
        seqs = model_dict['my_func9'](**params)
        return seqs
    params = params_dict['my_func10']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func10'](**params)
    params = params_dict['my_func10']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func10'](**params)
    params = params_dict['my_func10']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func10'](**params)
    params = params_dict['my_func10']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func10'](**params)
    params = params_dict['my_func10']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func10'](**params)
    params = params_dict['my_func10']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func10'](**params)
    params = params_dict['my_func10']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func10'](**params)
    params = params_dict['my_func10']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func10'](**params)
    params = params_dict['my_func10']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func10'](**params)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        params = params_dict['my_func11']
        input_dict = {}
        input_dict['seqs'] = seqs
        params.update(input_dict)
        seqs = model_dict['my_func11'](**params)
        return seqs
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func12'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func12'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func12'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func12'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func12'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func12'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func12'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func12'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func12'](**params)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        params = params_dict['my_func13']
        input_dict = {}
        input_dict['seqs'] = seqs
        params.update(input_dict)
        seqs = model_dict['my_func13'](**params)
        return seqs
    params = params_dict['my_func14']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func14'](**params)
    params = params_dict['my_func14']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func14'](**params)
    params = params_dict['my_func14']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func14'](**params)
    params = params_dict['my_func14']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func14'](**params)
    params = params_dict['my_func14']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func14'](**params)
    params = params_dict['my_func14']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func14'](**params)
    params = params_dict['my_func14']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func14'](**params)
    params = params_dict['my_func14']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func14'](**params)
    params = params_dict['my_func14']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func14'](**params)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        params = params_dict['my_func15']
        input_dict = {}
        input_dict['seqs'] = seqs
        params.update(input_dict)
        seqs = model_dict['my_func15'](**params)
        return seqs
    params = params_dict['my_func16']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, h, c, encoder_out, k_prev_words, next_word_inds) = model_dict['my_func16'](**params)
    params = params_dict['my_func16']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, h, c, encoder_out, k_prev_words, next_word_inds) = model_dict['my_func16'](**params)
    params = params_dict['my_func16']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, h, c, encoder_out, k_prev_words, next_word_inds) = model_dict['my_func16'](**params)
    params = params_dict['my_func16']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, h, c, encoder_out, k_prev_words, next_word_inds) = model_dict['my_func16'](**params)
    params = params_dict['my_func16']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, h, c, encoder_out, k_prev_words, next_word_inds) = model_dict['my_func16'](**params)
    params = params_dict['my_func16']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, h, c, encoder_out, k_prev_words, next_word_inds) = model_dict['my_func16'](**params)
    params = params_dict['my_func16']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, h, c, encoder_out, k_prev_words, next_word_inds) = model_dict['my_func16'](**params)
    params = params_dict['my_func16']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, h, c, encoder_out, k_prev_words, next_word_inds) = model_dict['my_func16'](**params)
    params = params_dict['my_func16']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, h, c, encoder_out, k_prev_words, next_word_inds) = model_dict['my_func16'](**params)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        params = params_dict['my_func17']
        input_dict = {}
        input_dict['seqs'] = seqs
        params.update(input_dict)
        seqs = model_dict['my_func17'](**params)
        return seqs
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['seqs'], input_dict['h'], input_dict['c'], input_dict['encoder_out'], input_dict['k_prev_words']) = (seqs, h, c, encoder_out, k_prev_words)
    params.update(input_dict)
    (seqs, next_word_inds) = model_dict['my_func18'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['seqs'], input_dict['h'], input_dict['c'], input_dict['encoder_out'], input_dict['k_prev_words']) = (seqs, h, c, encoder_out, k_prev_words)
    params.update(input_dict)
    (seqs, next_word_inds) = model_dict['my_func18'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['seqs'], input_dict['h'], input_dict['c'], input_dict['encoder_out'], input_dict['k_prev_words']) = (seqs, h, c, encoder_out, k_prev_words)
    params.update(input_dict)
    (seqs, next_word_inds) = model_dict['my_func18'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['seqs'], input_dict['h'], input_dict['c'], input_dict['encoder_out'], input_dict['k_prev_words']) = (seqs, h, c, encoder_out, k_prev_words)
    params.update(input_dict)
    (seqs, next_word_inds) = model_dict['my_func18'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['seqs'], input_dict['h'], input_dict['c'], input_dict['encoder_out'], input_dict['k_prev_words']) = (seqs, h, c, encoder_out, k_prev_words)
    params.update(input_dict)
    (seqs, next_word_inds) = model_dict['my_func18'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['seqs'], input_dict['h'], input_dict['c'], input_dict['encoder_out'], input_dict['k_prev_words']) = (seqs, h, c, encoder_out, k_prev_words)
    params.update(input_dict)
    (seqs, next_word_inds) = model_dict['my_func18'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['seqs'], input_dict['h'], input_dict['c'], input_dict['encoder_out'], input_dict['k_prev_words']) = (seqs, h, c, encoder_out, k_prev_words)
    params.update(input_dict)
    (seqs, next_word_inds) = model_dict['my_func18'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['seqs'], input_dict['h'], input_dict['c'], input_dict['encoder_out'], input_dict['k_prev_words']) = (seqs, h, c, encoder_out, k_prev_words)
    params.update(input_dict)
    (seqs, next_word_inds) = model_dict['my_func18'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['seqs'], input_dict['h'], input_dict['c'], input_dict['encoder_out'], input_dict['k_prev_words']) = (seqs, h, c, encoder_out, k_prev_words)
    params.update(input_dict)
    (seqs, next_word_inds) = model_dict['my_func18'](**params)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        params = params_dict['my_func19']
        input_dict = {}
        input_dict['seqs'] = seqs
        params.update(input_dict)
        seqs = model_dict['my_func19'](**params)
        return seqs
    params = params_dict['my_func20']
    input_dict = {}
    input_dict['seqs'] = seqs
    params.update(input_dict)
    seqs = model_dict['my_func20'](**params)
    return seqs



def TVM_API_Binary_No_OPT(input_dict, model_dict, self, constant_dict):
    m = model_dict['my_func0']
    m.set_input('input::x', input_dict['x'])
    m.run()
    seqs = m.get_output(0)
    encoder_out = m.get_output(1)
    k_prev_words = m.get_output(2)
    h = m.get_output(3)
    c = m.get_output(4)
    next_word_inds = m.get_output(5)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        m = model_dict['my_func1']
        m.set_input('input::seqs', seqs)
        m.run()
        seqs = m.get_output(0)
        return seqs
    m = model_dict['my_func2']
    m.set_input('input::encoder_out', encoder_out)
    m.set_input('input::k_prev_words', k_prev_words)
    m.set_input('input::seqs', seqs)
    m.set_input('input::h', h)
    m.set_input('input::c', c)
    m.run()
    seqs = m.get_output(0)
    encoder_out = m.get_output(1)
    k_prev_words = m.get_output(2)
    h = m.get_output(3)
    c = m.get_output(4)
    next_word_inds = m.get_output(5)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        m = model_dict['my_func3']
        m.set_input('input::seqs', seqs)
        m.run()
        seqs = m.get_output(0)
        return seqs
    m = model_dict['my_func4']
    m.set_input('input::encoder_out', encoder_out)
    m.set_input('input::k_prev_words', k_prev_words)
    m.set_input('input::seqs', seqs)
    m.set_input('input::h', h)
    m.set_input('input::c', c)
    m.run()
    seqs = m.get_output(0)
    encoder_out = m.get_output(1)
    k_prev_words = m.get_output(2)
    h = m.get_output(3)
    c = m.get_output(4)
    next_word_inds = m.get_output(5)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        m = model_dict['my_func5']
        m.set_input('input::seqs', seqs)
        m.run()
        seqs = m.get_output(0)
        return seqs
    m = model_dict['my_func6']
    m.set_input('input::encoder_out', encoder_out)
    m.set_input('input::k_prev_words', k_prev_words)
    m.set_input('input::seqs', seqs)
    m.set_input('input::h', h)
    m.set_input('input::c', c)
    m.run()
    seqs = m.get_output(0)
    encoder_out = m.get_output(1)
    k_prev_words = m.get_output(2)
    h = m.get_output(3)
    c = m.get_output(4)
    next_word_inds = m.get_output(5)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        m = model_dict['my_func7']
        m.set_input('input::seqs', seqs)
        m.run()
        seqs = m.get_output(0)
        return seqs
    m = model_dict['my_func8']
    m.set_input('input::encoder_out', encoder_out)
    m.set_input('input::k_prev_words', k_prev_words)
    m.set_input('input::seqs', seqs)
    m.set_input('input::h', h)
    m.set_input('input::c', c)
    m.run()
    seqs = m.get_output(0)
    encoder_out = m.get_output(1)
    k_prev_words = m.get_output(2)
    h = m.get_output(3)
    c = m.get_output(4)
    next_word_inds = m.get_output(5)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        m = model_dict['my_func9']
        m.set_input('input::seqs', seqs)
        m.run()
        seqs = m.get_output(0)
        return seqs
    m = model_dict['my_func10']
    m.set_input('input::encoder_out', encoder_out)
    m.set_input('input::k_prev_words', k_prev_words)
    m.set_input('input::seqs', seqs)
    m.set_input('input::h', h)
    m.set_input('input::c', c)
    m.run()
    seqs = m.get_output(0)
    encoder_out = m.get_output(1)
    k_prev_words = m.get_output(2)
    h = m.get_output(3)
    c = m.get_output(4)
    next_word_inds = m.get_output(5)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        m = model_dict['my_func11']
        m.set_input('input::seqs', seqs)
        m.run()
        seqs = m.get_output(0)
        return seqs
    m = model_dict['my_func12']
    m.set_input('input::encoder_out', encoder_out)
    m.set_input('input::k_prev_words', k_prev_words)
    m.set_input('input::seqs', seqs)
    m.set_input('input::h', h)
    m.set_input('input::c', c)
    m.run()
    seqs = m.get_output(0)
    encoder_out = m.get_output(1)
    k_prev_words = m.get_output(2)
    h = m.get_output(3)
    c = m.get_output(4)
    next_word_inds = m.get_output(5)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        m = model_dict['my_func13']
        m.set_input('input::seqs', seqs)
        m.run()
        seqs = m.get_output(0)
        return seqs
    m = model_dict['my_func14']
    m.set_input('input::encoder_out', encoder_out)
    m.set_input('input::k_prev_words', k_prev_words)
    m.set_input('input::seqs', seqs)
    m.set_input('input::h', h)
    m.set_input('input::c', c)
    m.run()
    seqs = m.get_output(0)
    encoder_out = m.get_output(1)
    k_prev_words = m.get_output(2)
    h = m.get_output(3)
    c = m.get_output(4)
    next_word_inds = m.get_output(5)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        m = model_dict['my_func15']
        m.set_input('input::seqs', seqs)
        m.run()
        seqs = m.get_output(0)
        return seqs
    m = model_dict['my_func16']
    m.set_input('input::encoder_out', encoder_out)
    m.set_input('input::k_prev_words', k_prev_words)
    m.set_input('input::seqs', seqs)
    m.set_input('input::h', h)
    m.set_input('input::c', c)
    m.run()
    seqs = m.get_output(0)
    h = m.get_output(1)
    c = m.get_output(2)
    encoder_out = m.get_output(3)
    k_prev_words = m.get_output(4)
    next_word_inds = m.get_output(5)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        m = model_dict['my_func17']
        m.set_input('input::seqs', seqs)
        m.run()
        seqs = m.get_output(0)
        return seqs
    m = model_dict['my_func18']
    m.set_input('input::seqs', seqs)
    m.set_input('input::h', h)
    m.set_input('input::c', c)
    m.set_input('input::encoder_out', encoder_out)
    m.set_input('input::k_prev_words', k_prev_words)
    m.run()
    seqs = m.get_output(0)
    next_word_inds = m.get_output(1)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        m = model_dict['my_func19']
        m.set_input('input::seqs', seqs)
        m.run()
        seqs = m.get_output(0)
        return seqs
    m = model_dict['my_func20']
    m.set_input('input::seqs', seqs)
    m.run()
    seqs = m.get_output(0)
    return seqs






def predictAPI_OPT(input_dict, model_dict, self, constant_dict):
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func0'](input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict['seqs'] = seqs
        seqs = model_dict['my_func1'](input_dict)
        return seqs
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func2'](input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict['seqs'] = seqs
        seqs = model_dict['my_func3'](input_dict)
        return seqs
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func4'](input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict['seqs'] = seqs
        seqs = model_dict['my_func5'](input_dict)
        return seqs
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func6'](input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict['seqs'] = seqs
        seqs = model_dict['my_func7'](input_dict)
        return seqs
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func8'](input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict['seqs'] = seqs
        seqs = model_dict['my_func9'](input_dict)
        return seqs
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func10'](input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict['seqs'] = seqs
        seqs = model_dict['my_func11'](input_dict)
        return seqs
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func12'](input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict['seqs'] = seqs
        seqs = model_dict['my_func13'](input_dict)
        return seqs
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func14'](input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict['seqs'] = seqs
        seqs = model_dict['my_func15'](input_dict)
        return seqs
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    (seqs, h, c, encoder_out, k_prev_words, next_word_inds) = model_dict['my_func16'](input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict['seqs'] = seqs
        seqs = model_dict['my_func17'](input_dict)
        return seqs
    (input_dict['seqs'], input_dict['h'], input_dict['c'], input_dict['encoder_out'], input_dict['k_prev_words']) = (seqs, h, c, encoder_out, k_prev_words)
    (seqs, next_word_inds) = model_dict['my_func18'](input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict['seqs'] = seqs
        seqs = model_dict['my_func19'](input_dict)
        return seqs
    input_dict['seqs'] = seqs
    seqs = model_dict['my_func20'](input_dict)
    return seqs



def ONNX_API_OPT(input_dict, model_dict, self, constant_dict):
    [seqs, encoder_out, k_prev_words, h, c, next_word_inds] = model_dict['my_func0'].run(['output::seqs', 'output::encoder_out', 'output::k_prev_words', 'output::h', 'output::c', 'output::next_word_inds'], input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict = {}
        input_dict['input::seqs'] = seqs
        [seqs] = model_dict['my_func1'].run(['output::seqs'], input_dict)
        return seqs
    input_dict = {}
    (input_dict['input::encoder_out'], input_dict['input::k_prev_words'], input_dict['input::seqs'], input_dict['input::h'], input_dict['input::c']) = (encoder_out, k_prev_words, seqs, h, c)
    [seqs, encoder_out, k_prev_words, h, c, next_word_inds] = model_dict['my_func2'].run(['output::seqs', 'output::encoder_out', 'output::k_prev_words', 'output::h', 'output::c', 'output::next_word_inds'], input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict = {}
        input_dict['input::seqs'] = seqs
        [seqs] = model_dict['my_func3'].run(['output::seqs'], input_dict)
        return seqs
    input_dict = {}
    (input_dict['input::encoder_out'], input_dict['input::k_prev_words'], input_dict['input::seqs'], input_dict['input::h'], input_dict['input::c']) = (encoder_out, k_prev_words, seqs, h, c)
    [seqs, encoder_out, k_prev_words, h, c, next_word_inds] = model_dict['my_func4'].run(['output::seqs', 'output::encoder_out', 'output::k_prev_words', 'output::h', 'output::c', 'output::next_word_inds'], input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict = {}
        input_dict['input::seqs'] = seqs
        [seqs] = model_dict['my_func5'].run(['output::seqs'], input_dict)
        return seqs
    input_dict = {}
    (input_dict['input::encoder_out'], input_dict['input::k_prev_words'], input_dict['input::seqs'], input_dict['input::h'], input_dict['input::c']) = (encoder_out, k_prev_words, seqs, h, c)
    [seqs, encoder_out, k_prev_words, h, c, next_word_inds] = model_dict['my_func6'].run(['output::seqs', 'output::encoder_out', 'output::k_prev_words', 'output::h', 'output::c', 'output::next_word_inds'], input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict = {}
        input_dict['input::seqs'] = seqs
        [seqs] = model_dict['my_func7'].run(['output::seqs'], input_dict)
        return seqs
    input_dict = {}
    (input_dict['input::encoder_out'], input_dict['input::k_prev_words'], input_dict['input::seqs'], input_dict['input::h'], input_dict['input::c']) = (encoder_out, k_prev_words, seqs, h, c)
    [seqs, encoder_out, k_prev_words, h, c, next_word_inds] = model_dict['my_func8'].run(['output::seqs', 'output::encoder_out', 'output::k_prev_words', 'output::h', 'output::c', 'output::next_word_inds'], input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict = {}
        input_dict['input::seqs'] = seqs
        [seqs] = model_dict['my_func9'].run(['output::seqs'], input_dict)
        return seqs
    input_dict = {}
    (input_dict['input::encoder_out'], input_dict['input::k_prev_words'], input_dict['input::seqs'], input_dict['input::h'], input_dict['input::c']) = (encoder_out, k_prev_words, seqs, h, c)
    [seqs, encoder_out, k_prev_words, h, c, next_word_inds] = model_dict['my_func10'].run(['output::seqs', 'output::encoder_out', 'output::k_prev_words', 'output::h', 'output::c', 'output::next_word_inds'], input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict = {}
        input_dict['input::seqs'] = seqs
        [seqs] = model_dict['my_func11'].run(['output::seqs'], input_dict)
        return seqs
    input_dict = {}
    (input_dict['input::encoder_out'], input_dict['input::k_prev_words'], input_dict['input::seqs'], input_dict['input::h'], input_dict['input::c']) = (encoder_out, k_prev_words, seqs, h, c)
    [seqs, encoder_out, k_prev_words, h, c, next_word_inds] = model_dict['my_func12'].run(['output::seqs', 'output::encoder_out', 'output::k_prev_words', 'output::h', 'output::c', 'output::next_word_inds'], input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict = {}
        input_dict['input::seqs'] = seqs
        [seqs] = model_dict['my_func13'].run(['output::seqs'], input_dict)
        return seqs
    input_dict = {}
    (input_dict['input::encoder_out'], input_dict['input::k_prev_words'], input_dict['input::seqs'], input_dict['input::h'], input_dict['input::c']) = (encoder_out, k_prev_words, seqs, h, c)
    [seqs, encoder_out, k_prev_words, h, c, next_word_inds] = model_dict['my_func14'].run(['output::seqs', 'output::encoder_out', 'output::k_prev_words', 'output::h', 'output::c', 'output::next_word_inds'], input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict = {}
        input_dict['input::seqs'] = seqs
        [seqs] = model_dict['my_func15'].run(['output::seqs'], input_dict)
        return seqs
    input_dict = {}
    (input_dict['input::encoder_out'], input_dict['input::k_prev_words'], input_dict['input::seqs'], input_dict['input::h'], input_dict['input::c']) = (encoder_out, k_prev_words, seqs, h, c)
    [seqs, h, c, encoder_out, k_prev_words, next_word_inds] = model_dict['my_func16'].run(['output::seqs', 'output::h', 'output::c', 'output::encoder_out', 'output::k_prev_words', 'output::next_word_inds'], input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict = {}
        input_dict['input::seqs'] = seqs
        [seqs] = model_dict['my_func17'].run(['output::seqs'], input_dict)
        return seqs
    input_dict = {}
    (input_dict['input::seqs'], input_dict['input::h'], input_dict['input::c'], input_dict['input::encoder_out'], input_dict['input::k_prev_words']) = (seqs, h, c, encoder_out, k_prev_words)
    [seqs, next_word_inds] = model_dict['my_func18'].run(['output::seqs', 'output::next_word_inds'], input_dict)
    if next_word_inds[0] == self.word_map['<end>']:
        input_dict = {}
        input_dict['input::seqs'] = seqs
        [seqs] = model_dict['my_func19'].run(['output::seqs'], input_dict)
        return seqs
    input_dict = {}
    input_dict['input::seqs'] = seqs
    [seqs] = model_dict['my_func20'].run(['output::seqs'], input_dict)
    return seqs



def TVM_API_OPT(input_dict, model_dict, params_dict, self, constant_dict):
    params = params_dict['my_func0']
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func0'](**params)
    params = params_dict['my_func0']
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func0'](**params)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        params = params_dict['my_func1']
        input_dict = {}
        input_dict['seqs'] = seqs
        params.update(input_dict)
        seqs = model_dict['my_func1'](**params)
        return seqs
    params = params_dict['my_func2']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func2'](**params)
    params = params_dict['my_func2']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func2'](**params)
    params = params_dict['my_func2']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func2'](**params)
    params = params_dict['my_func2']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func2'](**params)
    params = params_dict['my_func2']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func2'](**params)
    params = params_dict['my_func2']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func2'](**params)
    params = params_dict['my_func2']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func2'](**params)
    params = params_dict['my_func2']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func2'](**params)
    params = params_dict['my_func2']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func2'](**params)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        params = params_dict['my_func3']
        input_dict = {}
        input_dict['seqs'] = seqs
        params.update(input_dict)
        seqs = model_dict['my_func3'](**params)
        return seqs
    params = params_dict['my_func4']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func4'](**params)
    params = params_dict['my_func4']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func4'](**params)
    params = params_dict['my_func4']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func4'](**params)
    params = params_dict['my_func4']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func4'](**params)
    params = params_dict['my_func4']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func4'](**params)
    params = params_dict['my_func4']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func4'](**params)
    params = params_dict['my_func4']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func4'](**params)
    params = params_dict['my_func4']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func4'](**params)
    params = params_dict['my_func4']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func4'](**params)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        params = params_dict['my_func5']
        input_dict = {}
        input_dict['seqs'] = seqs
        params.update(input_dict)
        seqs = model_dict['my_func5'](**params)
        return seqs
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func6'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func6'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func6'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func6'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func6'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func6'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func6'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func6'](**params)
    params = params_dict['my_func6']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func6'](**params)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        params = params_dict['my_func7']
        input_dict = {}
        input_dict['seqs'] = seqs
        params.update(input_dict)
        seqs = model_dict['my_func7'](**params)
        return seqs
    params = params_dict['my_func8']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func8'](**params)
    params = params_dict['my_func8']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func8'](**params)
    params = params_dict['my_func8']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func8'](**params)
    params = params_dict['my_func8']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func8'](**params)
    params = params_dict['my_func8']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func8'](**params)
    params = params_dict['my_func8']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func8'](**params)
    params = params_dict['my_func8']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func8'](**params)
    params = params_dict['my_func8']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func8'](**params)
    params = params_dict['my_func8']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func8'](**params)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        params = params_dict['my_func9']
        input_dict = {}
        input_dict['seqs'] = seqs
        params.update(input_dict)
        seqs = model_dict['my_func9'](**params)
        return seqs
    params = params_dict['my_func10']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func10'](**params)
    params = params_dict['my_func10']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func10'](**params)
    params = params_dict['my_func10']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func10'](**params)
    params = params_dict['my_func10']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func10'](**params)
    params = params_dict['my_func10']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func10'](**params)
    params = params_dict['my_func10']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func10'](**params)
    params = params_dict['my_func10']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func10'](**params)
    params = params_dict['my_func10']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func10'](**params)
    params = params_dict['my_func10']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func10'](**params)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        params = params_dict['my_func11']
        input_dict = {}
        input_dict['seqs'] = seqs
        params.update(input_dict)
        seqs = model_dict['my_func11'](**params)
        return seqs
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func12'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func12'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func12'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func12'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func12'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func12'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func12'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func12'](**params)
    params = params_dict['my_func12']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func12'](**params)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        params = params_dict['my_func13']
        input_dict = {}
        input_dict['seqs'] = seqs
        params.update(input_dict)
        seqs = model_dict['my_func13'](**params)
        return seqs
    params = params_dict['my_func14']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func14'](**params)
    params = params_dict['my_func14']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func14'](**params)
    params = params_dict['my_func14']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func14'](**params)
    params = params_dict['my_func14']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func14'](**params)
    params = params_dict['my_func14']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func14'](**params)
    params = params_dict['my_func14']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func14'](**params)
    params = params_dict['my_func14']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func14'](**params)
    params = params_dict['my_func14']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func14'](**params)
    params = params_dict['my_func14']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, encoder_out, k_prev_words, h, c, next_word_inds) = model_dict['my_func14'](**params)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        params = params_dict['my_func15']
        input_dict = {}
        input_dict['seqs'] = seqs
        params.update(input_dict)
        seqs = model_dict['my_func15'](**params)
        return seqs
    params = params_dict['my_func16']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, h, c, encoder_out, k_prev_words, next_word_inds) = model_dict['my_func16'](**params)
    params = params_dict['my_func16']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, h, c, encoder_out, k_prev_words, next_word_inds) = model_dict['my_func16'](**params)
    params = params_dict['my_func16']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, h, c, encoder_out, k_prev_words, next_word_inds) = model_dict['my_func16'](**params)
    params = params_dict['my_func16']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, h, c, encoder_out, k_prev_words, next_word_inds) = model_dict['my_func16'](**params)
    params = params_dict['my_func16']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, h, c, encoder_out, k_prev_words, next_word_inds) = model_dict['my_func16'](**params)
    params = params_dict['my_func16']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, h, c, encoder_out, k_prev_words, next_word_inds) = model_dict['my_func16'](**params)
    params = params_dict['my_func16']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, h, c, encoder_out, k_prev_words, next_word_inds) = model_dict['my_func16'](**params)
    params = params_dict['my_func16']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, h, c, encoder_out, k_prev_words, next_word_inds) = model_dict['my_func16'](**params)
    params = params_dict['my_func16']
    input_dict = {}
    (input_dict['encoder_out'], input_dict['k_prev_words'], input_dict['seqs'], input_dict['h'], input_dict['c']) = (encoder_out, k_prev_words, seqs, h, c)
    params.update(input_dict)
    (seqs, h, c, encoder_out, k_prev_words, next_word_inds) = model_dict['my_func16'](**params)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        params = params_dict['my_func17']
        input_dict = {}
        input_dict['seqs'] = seqs
        params.update(input_dict)
        seqs = model_dict['my_func17'](**params)
        return seqs
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['seqs'], input_dict['h'], input_dict['c'], input_dict['encoder_out'], input_dict['k_prev_words']) = (seqs, h, c, encoder_out, k_prev_words)
    params.update(input_dict)
    (seqs, next_word_inds) = model_dict['my_func18'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['seqs'], input_dict['h'], input_dict['c'], input_dict['encoder_out'], input_dict['k_prev_words']) = (seqs, h, c, encoder_out, k_prev_words)
    params.update(input_dict)
    (seqs, next_word_inds) = model_dict['my_func18'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['seqs'], input_dict['h'], input_dict['c'], input_dict['encoder_out'], input_dict['k_prev_words']) = (seqs, h, c, encoder_out, k_prev_words)
    params.update(input_dict)
    (seqs, next_word_inds) = model_dict['my_func18'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['seqs'], input_dict['h'], input_dict['c'], input_dict['encoder_out'], input_dict['k_prev_words']) = (seqs, h, c, encoder_out, k_prev_words)
    params.update(input_dict)
    (seqs, next_word_inds) = model_dict['my_func18'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['seqs'], input_dict['h'], input_dict['c'], input_dict['encoder_out'], input_dict['k_prev_words']) = (seqs, h, c, encoder_out, k_prev_words)
    params.update(input_dict)
    (seqs, next_word_inds) = model_dict['my_func18'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['seqs'], input_dict['h'], input_dict['c'], input_dict['encoder_out'], input_dict['k_prev_words']) = (seqs, h, c, encoder_out, k_prev_words)
    params.update(input_dict)
    (seqs, next_word_inds) = model_dict['my_func18'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['seqs'], input_dict['h'], input_dict['c'], input_dict['encoder_out'], input_dict['k_prev_words']) = (seqs, h, c, encoder_out, k_prev_words)
    params.update(input_dict)
    (seqs, next_word_inds) = model_dict['my_func18'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['seqs'], input_dict['h'], input_dict['c'], input_dict['encoder_out'], input_dict['k_prev_words']) = (seqs, h, c, encoder_out, k_prev_words)
    params.update(input_dict)
    (seqs, next_word_inds) = model_dict['my_func18'](**params)
    params = params_dict['my_func18']
    input_dict = {}
    (input_dict['seqs'], input_dict['h'], input_dict['c'], input_dict['encoder_out'], input_dict['k_prev_words']) = (seqs, h, c, encoder_out, k_prev_words)
    params.update(input_dict)
    (seqs, next_word_inds) = model_dict['my_func18'](**params)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        params = params_dict['my_func19']
        input_dict = {}
        input_dict['seqs'] = seqs
        params.update(input_dict)
        seqs = model_dict['my_func19'](**params)
        return seqs
    params = params_dict['my_func20']
    input_dict = {}
    input_dict['seqs'] = seqs
    params.update(input_dict)
    seqs = model_dict['my_func20'](**params)
    return seqs



def TVM_API_Binary_OPT(input_dict, model_dict, self, constant_dict):
    m = model_dict['my_func0']
    m.set_input('input::x', input_dict['x'])
    m.run()
    seqs = m.get_output(0)
    encoder_out = m.get_output(1)
    k_prev_words = m.get_output(2)
    h = m.get_output(3)
    c = m.get_output(4)
    next_word_inds = m.get_output(5)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        m = model_dict['my_func1']
        m.set_input('input::seqs', seqs)
        m.run()
        seqs = m.get_output(0)
        return seqs
    m = model_dict['my_func2']
    m.set_input('input::encoder_out', encoder_out)
    m.set_input('input::k_prev_words', k_prev_words)
    m.set_input('input::seqs', seqs)
    m.set_input('input::h', h)
    m.set_input('input::c', c)
    m.run()
    seqs = m.get_output(0)
    encoder_out = m.get_output(1)
    k_prev_words = m.get_output(2)
    h = m.get_output(3)
    c = m.get_output(4)
    next_word_inds = m.get_output(5)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        m = model_dict['my_func3']
        m.set_input('input::seqs', seqs)
        m.run()
        seqs = m.get_output(0)
        return seqs
    m = model_dict['my_func4']
    m.set_input('input::encoder_out', encoder_out)
    m.set_input('input::k_prev_words', k_prev_words)
    m.set_input('input::seqs', seqs)
    m.set_input('input::h', h)
    m.set_input('input::c', c)
    m.run()
    seqs = m.get_output(0)
    encoder_out = m.get_output(1)
    k_prev_words = m.get_output(2)
    h = m.get_output(3)
    c = m.get_output(4)
    next_word_inds = m.get_output(5)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        m = model_dict['my_func5']
        m.set_input('input::seqs', seqs)
        m.run()
        seqs = m.get_output(0)
        return seqs
    m = model_dict['my_func6']
    m.set_input('input::encoder_out', encoder_out)
    m.set_input('input::k_prev_words', k_prev_words)
    m.set_input('input::seqs', seqs)
    m.set_input('input::h', h)
    m.set_input('input::c', c)
    m.run()
    seqs = m.get_output(0)
    encoder_out = m.get_output(1)
    k_prev_words = m.get_output(2)
    h = m.get_output(3)
    c = m.get_output(4)
    next_word_inds = m.get_output(5)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        m = model_dict['my_func7']
        m.set_input('input::seqs', seqs)
        m.run()
        seqs = m.get_output(0)
        return seqs
    m = model_dict['my_func8']
    m.set_input('input::encoder_out', encoder_out)
    m.set_input('input::k_prev_words', k_prev_words)
    m.set_input('input::seqs', seqs)
    m.set_input('input::h', h)
    m.set_input('input::c', c)
    m.run()
    seqs = m.get_output(0)
    encoder_out = m.get_output(1)
    k_prev_words = m.get_output(2)
    h = m.get_output(3)
    c = m.get_output(4)
    next_word_inds = m.get_output(5)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        m = model_dict['my_func9']
        m.set_input('input::seqs', seqs)
        m.run()
        seqs = m.get_output(0)
        return seqs
    m = model_dict['my_func10']
    m.set_input('input::encoder_out', encoder_out)
    m.set_input('input::k_prev_words', k_prev_words)
    m.set_input('input::seqs', seqs)
    m.set_input('input::h', h)
    m.set_input('input::c', c)
    m.run()
    seqs = m.get_output(0)
    encoder_out = m.get_output(1)
    k_prev_words = m.get_output(2)
    h = m.get_output(3)
    c = m.get_output(4)
    next_word_inds = m.get_output(5)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        m = model_dict['my_func11']
        m.set_input('input::seqs', seqs)
        m.run()
        seqs = m.get_output(0)
        return seqs
    m = model_dict['my_func12']
    m.set_input('input::encoder_out', encoder_out)
    m.set_input('input::k_prev_words', k_prev_words)
    m.set_input('input::seqs', seqs)
    m.set_input('input::h', h)
    m.set_input('input::c', c)
    m.run()
    seqs = m.get_output(0)
    encoder_out = m.get_output(1)
    k_prev_words = m.get_output(2)
    h = m.get_output(3)
    c = m.get_output(4)
    next_word_inds = m.get_output(5)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        m = model_dict['my_func13']
        m.set_input('input::seqs', seqs)
        m.run()
        seqs = m.get_output(0)
        return seqs
    m = model_dict['my_func14']
    m.set_input('input::encoder_out', encoder_out)
    m.set_input('input::k_prev_words', k_prev_words)
    m.set_input('input::seqs', seqs)
    m.set_input('input::h', h)
    m.set_input('input::c', c)
    m.run()
    seqs = m.get_output(0)
    encoder_out = m.get_output(1)
    k_prev_words = m.get_output(2)
    h = m.get_output(3)
    c = m.get_output(4)
    next_word_inds = m.get_output(5)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        m = model_dict['my_func15']
        m.set_input('input::seqs', seqs)
        m.run()
        seqs = m.get_output(0)
        return seqs
    m = model_dict['my_func16']
    m.set_input('input::encoder_out', encoder_out)
    m.set_input('input::k_prev_words', k_prev_words)
    m.set_input('input::seqs', seqs)
    m.set_input('input::h', h)
    m.set_input('input::c', c)
    m.run()
    seqs = m.get_output(0)
    h = m.get_output(1)
    c = m.get_output(2)
    encoder_out = m.get_output(3)
    k_prev_words = m.get_output(4)
    next_word_inds = m.get_output(5)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        m = model_dict['my_func17']
        m.set_input('input::seqs', seqs)
        m.run()
        seqs = m.get_output(0)
        return seqs
    m = model_dict['my_func18']
    m.set_input('input::seqs', seqs)
    m.set_input('input::h', h)
    m.set_input('input::c', c)
    m.set_input('input::encoder_out', encoder_out)
    m.set_input('input::k_prev_words', k_prev_words)
    m.run()
    seqs = m.get_output(0)
    next_word_inds = m.get_output(1)
    if next_word_inds.asnumpy()[0] == self.word_map['<end>']:
        m = model_dict['my_func19']
        m.set_input('input::seqs', seqs)
        m.run()
        seqs = m.get_output(0)
        return seqs
    m = model_dict['my_func20']
    m.set_input('input::seqs', seqs)
    m.run()
    seqs = m.get_output(0)
    return seqs



