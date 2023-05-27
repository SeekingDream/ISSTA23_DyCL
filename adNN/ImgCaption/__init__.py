import torch
from torch.utils.data import Dataset
import h5py
import json
from operator import itemgetter
import os
import torch.nn as nn

from .model import *


MODEL_FILE_LIST = [
    # 'flickr8k_googlenet_rnn',
    'flickr8k_resnext_lstm',
    # 'coco_mobilenet_rnn',
    'coco_resnet_lstm',
]


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split == 'TRAIN':
            return img
            # return img, caption, caplen
        else:
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            # return img, caption, caplen, all_captions
            return img

    def __len__(self):
        return self.dataset_size


class CaptionModel(nn.Module):
    def __init__(self, encoder, decoder, word_map, max_length):
        super(CaptionModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.word_map = word_map
        self.max_length = max_length

    def forward(self, imgs):
        encoder_out = self.encoder(imgs)  # (1, enc_image_size, enc_image_size, encoder_dim)
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (1, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[self.word_map['<start>']]] * batch_size) # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        h, c = self.decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        for i in range(self.max_length):
            embeddings = self.decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, alpha = self.decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            gate = self.decoder.sigmoid(self.decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = self.decoder.fc(h)  # (s, vocab_size)
            _, next_word_inds = scores.max(1)
            next_word_inds = next_word_inds.cpu()

            seqs = torch.cat([seqs, next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            if next_word_inds[0] == self.word_map['<end>']:
                return seqs

        return seqs

    def forward_compile(self, x):
        encoder_out = self.encoder(x)  # (1, enc_image_size, enc_image_size, encoder_dim)
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (1, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[self.word_map['<start>']]] * batch_size).to(x.device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        h, c = self.decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        for i in range(self.max_length):
            embeddings = self.decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, alpha = self.decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            gate = self.decoder.sigmoid(self.decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = self.decoder.fc(h)  # (s, vocab_size)
            _, next_word_inds = scores.max(1)

            seqs = torch.cat([seqs, next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            if next_word_inds[0] == self.word_map['<end>']:
                return seqs
        return seqs


def create_encoder_decoder(task_id, word_map):
    device = torch.device('cuda')

    config_name = MODEL_FILE_LIST[task_id] + '.json'
    with open(os.path.join('./adNN/ImgCaption/config', config_name), 'r') as f:
        config = json.load(f)
    encoder_type = config['model']['encoder']
    encoder_dim = config['model']["encoder_dim"]
    emb_dim = config['model']['emb_dim']  # dimension of word embeddings
    attention_dim = config['model']['attention_dim']  # dimension of attention linear layers
    decoder_dim = config['model']['decoder_dim']  # dimension of decoder RNN
    dropout = config['model']['dropout']
    decoder_type = config['model']['decoder']
    data_name = config['data']['data_name']
    dataset_name = MODEL_FILE_LIST[task_id].split('_')[0]

    decoder = get_decoder(
        decoder_type=decoder_type,
        attention_dim=attention_dim,
        embed_dim=emb_dim,
        decoder_dim=decoder_dim,
        vocab_size=len(word_map),
        device=device,
        encoder_dim=encoder_dim,
        dropout=dropout)
    encoder = Encoder(encoder_type=encoder_type)

    return encoder, decoder


def load_encoder_decoder(data_path, task_id, batch_size=1, num_workers=1):
    data_name = MODEL_FILE_LIST[task_id].split('_')[0]
    data_folder = os.path.join(data_path, data_name)

    DATA_NAME_DICT = {
        'coco': 'coco_5_cap_per_img_3_min_word_freq',
        'flickr8k': 'flickr8k_5_cap_per_img_5_min_word_freq'
    }

    task_name = DATA_NAME_DICT[data_name]

    # test_loader = torch.utils.data.DataLoader(
    #     CaptionDataset(data_folder, task_name, 'TEST', ),
    #     batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    word_map_path = os.path.join(data_path, task_name.split('_')[0] + '/WORDMAP_' + task_name + '.json')
    with open(word_map_path, 'r') as j:
        word_map = json.load(j)

    encoder, decoder = create_encoder_decoder(task_id, word_map)
    # encoder.load_state_dict(encoder_s)
    # decoder.load_state_dict(decoder_s)
    test_loader = None
    return encoder, decoder, test_loader, word_map


def load_img_caption_model(data_path, task_id, max_length):
    encoder, decoder, test_loader, word_map = load_encoder_decoder(data_path, task_id)
    caption_model = CaptionModel(encoder, decoder, word_map, max_length=max_length)
    return caption_model, test_loader

