"""
@author: Gaetan Hadjeres
"""

import random
import numpy as np
import torch
from DatasetManager.chorale_dataset import ChoraleDataset
from DeepBach.helpers import cuda_variable, init_hidden

from torch import nn

from DeepBach.data_utils import reverse_tensor, mask_entry

def get_c_kernel(n):
    # n X 64 X 32
    e = 64
    activ_fn = nn.LeakyReLU()
    res = nn.Sequential(
        nn.Conv2d(n, 16, kernel_size=(5, 5)), # 16 X 60 X 28
        activ_fn,
        nn.MaxPool2d((2, 2), stride=2), # 16 X 30 X 14
        nn.Conv2d(16, 32, kernel_size=(5, 5), padding=(1, 1)), # 32 X 28 X 12
        activ_fn,
        nn.MaxPool2d((2, 2), stride=2), # 32 X 14 X 6
        nn.Conv2d(32, 64, kernel_size=(3, 3)), # 64 X 12 X 4
        activ_fn,
        nn.MaxPool2d((2, 2), stride=1), # 64 X 11 X 3
        nn.Conv2d(64, n*e, kernel_size=(5, 3), stride=(3, 1)), # n*e X 3 X 1
        activ_fn,
    )
    return res

class VoiceModel(nn.Module):
    def __init__(self,
                 dataset: ChoraleDataset,
                 main_voice_index: int,
                 note_embedding_dim: int,
                 meta_embedding_dim: int,
                 num_layers: int,
                 lstm_hidden_size: int,
                 dropout_lstm: float,
                 hidden_size_linear=200
                 ):
        super(VoiceModel, self).__init__()
        assert note_embedding_dim == meta_embedding_dim
        self.dataset = dataset
        self.main_voice_index = main_voice_index
        self.note_embedding_dim = note_embedding_dim
        self.meta_embedding_dim = meta_embedding_dim
        self.num_notes_per_voice = [len(d)
                                    for d in dataset.note2index_dicts]
        self.num_voices = self.dataset.num_voices
        self.num_metas_per_voice = [
                                       metadata.num_values
                                       for metadata in dataset.metadatas
                                   ] + [self.num_voices]
        self.num_metas = len(self.dataset.metadatas) + 1
        self.num_layers = num_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.dropout_lstm = dropout_lstm
        self.hidden_size_linear = hidden_size_linear

        self.other_voices_indexes = [i
                                     for i
                                     in range(self.num_voices)
                                     if not i == main_voice_index]

        self.note_embeddings = nn.ModuleList(
            [nn.Embedding(num_notes, note_embedding_dim)
             for num_notes in self.num_notes_per_voice]
        )
        self.meta_embeddings = nn.ModuleList(
            [nn.Embedding(num_metas, meta_embedding_dim)
             for num_metas in self.num_metas_per_voice]
        )

        self.lstm_left = nn.LSTM(input_size=note_embedding_dim * self.num_voices +
                                            meta_embedding_dim * self.num_metas,
                                 hidden_size=lstm_hidden_size,
                                 num_layers=num_layers,
                                 dropout=dropout_lstm,
                                 batch_first=True)
        self.lstm_right = nn.LSTM(input_size=note_embedding_dim * self.num_voices +
                                             meta_embedding_dim * self.num_metas,
                                  hidden_size=lstm_hidden_size,
                                  num_layers=num_layers,
                                  dropout=dropout_lstm,
                                  batch_first=True)

        # self.attn = nn.Linear(self.hidden_size * 2, self.max_length)

        # ned*(2nv + 1)+med*nm*2 X ntsteps
        # ksize = 
        self.emb_size = (note_embedding_dim * (self.num_voices - 1)*2 + note_embedding_dim + (meta_embedding_dim * self.num_metas)*2)
        # self.conv = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=(11,3),padding=(5,1)), # 16 X 120 X 31,
        #     nn.ReLU(),
        #     nn.AvgPool2d((4,3), stride=2), # 16 X 59 X 15
        #     nn.Conv2d(16, 32, kernel_size=(5,3), padding=(2,1)), # 32 X 59 X 15
        #     nn.ReLU(),
        #     nn.MaxPool2d((3, 3), stride=3),  # 32 X 19 X 5
        #     nn.Conv2d(32, 64, kernel_size=(3,5)), # 64 X 17 X 1
        #     nn.ReLU(),
        #     # nn.AvgPool2d((4,3), stride=2), # 64 X 17 X 1
        #     nn.Conv2d(64, 60, kernel_size=(5, 1), stride=(4, 1)), # 60 X 4 X 1
        #     nn.ReLU(),
        # )
        self.conv = get_c_kernel(self.num_voices+self.num_metas)
        self.num_missing = 5

        # self.conv_glob = nn.Sequential(
        #     nn.Conv2d()
        # )

        self.attn = nn.Sequential(
            nn.Linear((note_embedding_dim * (self.num_voices - 1)*2 + note_embedding_dim
                       + (meta_embedding_dim * self.num_metas)*2),
                      hidden_size_linear),
            nn.ReLU(),
            nn.Linear(hidden_size_linear, 1)
        )

        self.mlp_center = nn.Sequential(
            nn.Linear((note_embedding_dim * (self.num_voices - 1)
                       + meta_embedding_dim * self.num_metas),
                      hidden_size_linear),
            nn.ReLU(),
            nn.Linear(hidden_size_linear, (note_embedding_dim * (self.num_voices) 
                       + (meta_embedding_dim * self.num_metas)))
        )

        self.mlp_predictions = nn.Sequential(
            nn.Linear((note_embedding_dim * (self.num_voices) 
                       + (meta_embedding_dim * self.num_metas)) * 4,
                      hidden_size_linear),
            nn.ReLU(),
            nn.Linear(hidden_size_linear, self.num_notes_per_voice[main_voice_index])
        )

    def forward(self, *input):
        notes, metas = input
        batch_size, num_voices, timesteps_ticks = notes[0].size()

        # put time first
        ln, cn, rn = notes
        ln, rn = [t.transpose(1, 2)
                  for t in (ln, rn)]
        notes = ln, cn, rn

        # embedding
        notes_embedded = self.embed(notes, type='note')
        metas_embedded = self.embed(metas, type='meta')
        # lists of (N, timesteps_ticks, voices * dim_embedding)
        # where timesteps_ticks is 1 for central parts

        # concat notes and metas
        input_embedded = [torch.cat([notes, metas], 2) if notes is not None else None
                          for notes, metas in zip(notes_embedded, metas_embedded)]

        left, center, right = input_embedded

        # print("left.size(),right.size(),center.size(), self.num_voices, self.num_metas, self.note_embedding_dim, self.meta_embedding_dim")
        # print(left.size(),right.size(),center.size(), self.num_voices, self.num_metas, self.note_embedding_dim, self.meta_embedding_dim)
        

        ############## Attention ##############

        # strip -> bs X ntsteps X ndim
        bs=batch_size
        zs = torch.zeros(bs, 1, self.note_embedding_dim*(self.num_voices+self.num_metas)).cuda()
        strip = torch.cat([left, zs, right], 1)
        # center2 = torch.cat([center, torch.zeros(bs, 1, self.note_embedding_dim).cuda()], 2)
        # sandwich = torch.cat([strip, center2.repeat(1, strip.size(1), 1)], 2).permute(0, 2, 1)
        # ntsteps = sandwich.size(-1)
        ntsteps = left.size(1)+right.size(1)+1
        # print("ntsteps", ntsteps)
        # sandwich = torch.reshape(sandwich, (-1, 1, strip.size(2), ntsteps))
        # sandwich = torch.reshape(sandwich, (-1, 2, int(sandwich.size(1) // 2), ntsteps))
        # -1 X 2 X 120 X 31
        # bs X 2 X (ne*nv+me*nm) X (ntsteps=31)
        # bs X (2*(ne*nv+me*nm))
        # bs X (2*120)
        # emb_size = (ne*nv+me*nm) # 120
        strip2 = torch.reshape(strip, (-1, self.num_voices+self.num_metas, self.note_embedding_dim, ntsteps))
        # print("strip2", strip2.size())
        for layer_id in range(self.num_layers):
            num_change = self.num_missing
            num_change = random.randint(num_change // 2, int(num_change * 1.5))
            choices = np.random.choice(ntsteps,size=num_change,replace=False).tolist()
            for choice in choices:
                strip2[:, layer_id, :, choice] = torch.randn(batch_size, self.note_embedding_dim).cuda()

        # conv_out = self.conv(sandwich).view(-1, 240)
        conv_out = self.conv(strip2) # n*e X 3 X blah
        # print("conv_out", conv_out.size())
        conv_out = torch.mean(conv_out, -1).view(-1, conv_out.size(1)*conv_out.size(2))
        

        center = self.mlp_center(center)
        center = center.view(-1, center.size(2))
        # print("center", center.size())
        # exit()
        # left = left_attn_value
        # right = right_attn_value

        # print(center.size(),left.size(),right.size())
        predictions = torch.cat([conv_out, center], 1)
        # predictions = torch.cat([
        #     left, center, right
        # ], 1)
        # print(predictions.size())
        # exit()
        predictions = self.mlp_predictions(predictions)
        # print(predictions.size())

        # exit()

        return predictions

    def embed(self, notes_or_metas, type):
        if type == 'note':
            embeddings = self.note_embeddings
            embedding_dim = self.note_embedding_dim
            other_voices_indexes = self.other_voices_indexes
        elif type == 'meta':
            embeddings = self.meta_embeddings
            embedding_dim = self.meta_embedding_dim
            other_voices_indexes = range(self.num_metas)

        batch_size, timesteps_left_ticks, num_voices = notes_or_metas[0].size()
        batch_size, timesteps_right_ticks, num_voices = notes_or_metas[2].size()

        left, center, right = notes_or_metas
        # center has self.num_voices - 1 voices
        left_embedded = torch.cat([
            embeddings[voice_id](left[:, :, voice_id])[:, :, None, :]
            for voice_id in range(num_voices)
        ], 2)
        right_embedded = torch.cat([
            embeddings[voice_id](right[:, :, voice_id])[:, :, None, :]
            for voice_id in range(num_voices)
        ], 2)
        if self.num_voices == 1 and type == 'note':
            center_embedded = None
        else:
            center_embedded = torch.cat([
                embeddings[voice_id](center[:, k].unsqueeze(1))
                for k, voice_id in enumerate(other_voices_indexes)
            ], 1)
            center_embedded = center_embedded.view(batch_size,
                                                   1,
                                                   len(other_voices_indexes) * embedding_dim)

        # squeeze two last dimensions
        left_embedded = left_embedded.view(batch_size,
                                           timesteps_left_ticks,
                                           num_voices * embedding_dim)
        right_embedded = right_embedded.view(batch_size,
                                             timesteps_right_ticks,
                                             num_voices * embedding_dim)

        return left_embedded, center_embedded, right_embedded

    def save(self):
        torch.save(self.state_dict(), 'models/' + self.__repr__())
        print(f'Model {self.__repr__()} saved')

    def load(self):
        state_dict = torch.load('models/' + self.__repr__(),
                                map_location=lambda storage, loc: storage)
        print(f'Loading {self.__repr__()}')
        self.load_state_dict(state_dict)

    def __repr__(self):
        return f'VoiceModel(' \
               f'{self.dataset.__repr__()},' \
               f'{self.main_voice_index},' \
               f'{self.note_embedding_dim},' \
               f'{self.meta_embedding_dim},' \
               f'{self.num_layers},' \
               f'{self.lstm_hidden_size},' \
               f'{self.dropout_lstm},' \
               f'{self.hidden_size_linear}' \
               f')'

    def train_model(self,
                    batch_size=16,
                    num_epochs=10,
                    optimizer=None):
        for epoch in range(num_epochs):
            print(f'===Epoch {epoch}===')
            (dataloader_train,
             dataloader_val,
             dataloader_test) = self.dataset.data_loaders(
                batch_size=batch_size,
            )

            loss, acc = self.loss_and_acc(dataloader_train,
                                          optimizer=optimizer,
                                          phase='train')
            print(f'Training loss: {loss}')
            print(f'Training accuracy: {acc}')
            # writer.add_scalar('data/training_loss', loss, epoch)
            # writer.add_scalar('data/training_acc', acc, epoch)

            loss, acc = self.loss_and_acc(dataloader_val,
                                          optimizer=None,
                                          phase='test')
            print(f'Validation loss: {loss}')
            print(f'Validation accuracy: {acc}')
            self.save()

    def loss_and_acc(self, dataloader,
                     optimizer=None,
                     phase='train'):

        average_loss = 0
        average_acc = 0
        if phase == 'train':
            self.train()
        elif phase == 'eval' or phase == 'test':
            self.eval()
        else:
            raise NotImplementedError
        print(len(dataloader))
        # exit()
        i=0
        loss_function = torch.nn.CrossEntropyLoss()
        for tensor_chorale, tensor_metadata in dataloader:
            # print(i)
            # i=i+1
            # if(i==2):
            #     exit()
            # to Variable
            tensor_chorale = cuda_variable(tensor_chorale).long()
            tensor_metadata = cuda_variable(tensor_metadata).long()

            # preprocessing to put in the DeepBach format
            # see Fig. 4 in DeepBach paper:
            # https://arxiv.org/pdf/1612.01010.pdf
            notes, metas, label = self.preprocess_input(tensor_chorale,
                                                        tensor_metadata)

            weights = self.forward(notes, metas)
            # print("conv:  wts{}\tlbl{}".format(weights.size(), label.size()))
            # exit()

            loss = loss_function(weights, label)

            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            acc = self.accuracy(weights=weights,
                                target=label)

            average_loss += loss.item()
            average_acc += acc.item()

        average_loss /= len(dataloader)
        average_acc /= len(dataloader)
        return average_loss, average_acc

    def accuracy(self, weights, target):
        batch_size, = target.size()
        softmax = nn.Softmax(dim=1)(weights)
        pred = softmax.max(1)[1].type_as(target)
        num_corrects = (pred == target).float().sum()
        return num_corrects / batch_size * 100

    def preprocess_input(self, tensor_chorale, tensor_metadata):
        """
        :param tensor_chorale: (batch_size, num_voices, chorale_length_ticks)
        :param tensor_metadata: (batch_size, num_metadata, chorale_length_ticks)
        :return: (notes, metas, label) tuple
        where
        notes = (left_notes, central_notes, right_notes)
        metas = (left_metas, central_metas, right_metas)
        label = (batch_size)
        right_notes and right_metas are REVERSED (from right to left)
        """
        batch_size, num_voices, chorale_length_ticks = tensor_chorale.size()

        # random shift! Depends on the dataset
        offset = random.randint(0, self.dataset.subdivision)
        time_index_ticks = chorale_length_ticks // 2 + offset

        # split notes
        notes, label = self.preprocess_notes(tensor_chorale, time_index_ticks)
        metas = self.preprocess_metas(tensor_metadata, time_index_ticks)
        return notes, metas, label

    def preprocess_notes(self, tensor_chorale, time_index_ticks):
        """

        :param tensor_chorale: (batch_size, num_voices, chorale_length_ticks)
        :param time_index_ticks:
        :return:
        """
        batch_size, num_voices, _ = tensor_chorale.size()
        left_notes = tensor_chorale[:, :, :time_index_ticks]
        right_notes = reverse_tensor(
            tensor_chorale[:, :, time_index_ticks + 1:],
            dim=2)
        if self.num_voices == 1:
            central_notes = None
        else:
            central_notes = mask_entry(tensor_chorale[:, :, time_index_ticks],
                                       entry_index=self.main_voice_index,
                                       dim=1)
        label = tensor_chorale[:, self.main_voice_index, time_index_ticks]
        return (left_notes, central_notes, right_notes), label

    def preprocess_metas(self, tensor_metadata, time_index_ticks):
        """

        :param tensor_metadata: (batch_size, num_voices, chorale_length_ticks)
        :param time_index_ticks:
        :return:
        """

        left_metas = tensor_metadata[:, self.main_voice_index, :time_index_ticks, :]
        right_metas = reverse_tensor(
            tensor_metadata[:, self.main_voice_index, time_index_ticks + 1:, :],
            dim=1)
        central_metas = tensor_metadata[:, self.main_voice_index, time_index_ticks, :]
        return left_metas, central_metas, right_metas
