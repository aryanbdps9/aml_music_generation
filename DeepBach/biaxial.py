"""
@author: Gaetan Hadjeres
"""

import random

import torch
from DatasetManager.chorale_dataset import ChoraleDataset
from DeepBach.helpers import cuda_variable, init_hidden

from torch import nn

from DeepBach.data_utils import reverse_tensor, mask_entry


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

		self.time_lstm_left = nn.LSTM(input_size=1,
								  hidden_size=1,
								  num_layers=num_layers,
								  dropout=dropout_lstm,
								  batch_first=True)
		self.notes_lstm_left = nn.LSTM(input_size=1,
								  hidden_size=lstm_hidden_size,
								  num_layers=num_layers,
								  dropout=dropout_lstm,
								  batch_first=True
			)

		self.time_lstm_right = nn.LSTM(input_size=1,
								  hidden_size=1,
								  num_layers=num_layers,
								  dropout=dropout_lstm,
								  batch_first=True)
		self.notes_lstm_right = nn.LSTM(input_size=1,
								  hidden_size=lstm_hidden_size,
								  num_layers=num_layers,
								  dropout=dropout_lstm,
								  batch_first=True
			)

		# self.attn = nn.Linear(self.hidden_size * 2, self.max_length)


		# ned*(2nv + 1)+med*nm*2 X ntsteps
		# ksize = 
		self.emb_size = (note_embedding_dim * (self.num_voices - 1)*2 + note_embedding_dim + (meta_embedding_dim * self.num_metas)*2)
		# self.conv = nn.Sequential(
		# 	nn.Conv2d(2, 16, kernel_size=(11,3),padding=(5,1)), # 16 X 120 X 31,
		# 	nn.ReLU(),
		# 	nn.AvgPool2d((4,3), stride=2), # 16 X 59 X 15
		# 	nn.Conv2d(16, 32, kernel_size=(5,3), padding=(2,1)), # 32 X 59 X 15
		# 	nn.ReLU(),
		# 	nn.AvgPool2d((3, 3), stride=3),  # 32 X 19 X 5
		# 	nn.Conv2d(32, 64, kernel_size=(3,5)), # 64 X 17 X 1
		# 	nn.ReLU(),
		# 	# nn.AvgPool2d((4,3), stride=2), # 64 X 17 X 1
		# 	nn.Conv2d(64, 60, kernel_size=(5, 1), stride=(4, 1)), # 60 X 4 X 1
		# 	nn.ReLU(),
		# )

		# self.attn = nn.Sequential(
		# 	nn.Linear((note_embedding_dim * (self.num_voices - 1)*2 + note_embedding_dim
		# 			   + (meta_embedding_dim * self.num_metas)*2),
		# 			  hidden_size_linear),
		# 	nn.ReLU(),
		# 	nn.Linear(hidden_size_linear, 1)
		# )

		# embed_size = (note_embedding_dim * (self.num_voices)
		# 			   + (meta_embedding_dim * self.num_metas)) # final embed size of left,right

		# self.left_attn = Attention(embed_size)
		# self.right_attn = Attention(embed_size)

		self.mlp_center = nn.Sequential(
			nn.Linear((note_embedding_dim * (self.num_voices )
					   + meta_embedding_dim * self.num_metas),
					  hidden_size_linear),
			nn.ReLU(),
			nn.Linear(hidden_size_linear, (note_embedding_dim * (self.num_voices) 
					   + (meta_embedding_dim * self.num_metas)))
		)

		self.mlp_predictions = nn.Sequential(
            nn.Linear(self.lstm_hidden_size * 2,
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
		# get one hot encodings

		ln_encode = torch.eye(self.num_notes_per_voice[0],device='cuda:0')[ln]
		rn_encode = torch.eye(self.num_notes_per_voice[0],device='cuda:0')[rn]		

		# ln_encode-size = 256 x 19 x 1 x 67 
		# ln_encode-size = 256 x 67 
		States_ln = torch.zeros(ln_encode.size()[0], ln_encode.size()[3], device='cuda:0')
		for i in range(self.num_notes_per_voice[0]):
			time_i = ln_encode[: , : , 0 , i].reshape(ln_encode.size()[0],ln_encode.size()[1], 1)
			# print("size of time input", time_i.size())
			A = self.time_lstm_left(time_i)[0]
			# print(type(A),A.size())
			#print("output" , A)
			States_ln[:,i] = A[:, -1,:].reshape(batch_size)

		# print("output of time lstm" ,States_ln.size())
		States_ln=States_ln.reshape(States_ln.size()[0] , States_ln.size()[1] , 1)
		note_pred_ln = self.notes_lstm_left(States_ln)[0][:,-1,:]
		# print("size of notes", note_pred_ln.size())

		States_rn = torch.zeros(rn_encode.size()[0], rn_encode.size()[3], device='cuda:0')
		for i in range(self.num_notes_per_voice[0]):
			time_i = rn_encode[: , : , 0 , i].reshape(rn_encode.size()[0],rn_encode.size()[1], 1)
			# print("size of time input", time_i.size())
			A  = self.time_lstm_right(time_i)[0]
			# print(type(A),A.size())
			#print("output" , A)
			States_rn[:,i] = A[:, -1,:].reshape(batch_size)

		# print("output of time lstm" ,States_rn.size())
		States_rn=States_rn.reshape(States_rn.size()[0] , States_rn.size()[1] , 1)
		note_pred_rn = self.notes_lstm_right(States_rn)[0][:,-1,:]
		# print("size of notes", note_pred_rn.size())


		# exit()
		# get the time-duration data.
		# ln, cn, rn   = notes
		# ln_d , ln_n	 = rle(ln[0])
		# rn_d , rn_n  = rle(rn[0])

		# embedding
		# notes_embedded = self.embed((ln_d ,None,  ln_n), type='note')
		# metas_embedded = self.embed((rn_d ,None, rn_n), type='meta')
		# lists of (N, timesteps_ticks, voices * dim_embedding)
		# where timesteps_ticks is 1 for central parts

		# concat notes and metas
		# input_embedded = [torch.cat([notes, metas], 2) if notes is not None else None
		# 				  for notes, metas in zip(notes_embedded, metas_embedded)]



		# left, center, right = input_embedded
		

		
		# bs = batch_size
		# center = torch.cat([center, torch.zeros(bs, 1, self.note_embedding_dim).cuda()], 2)	# conv_out = self.conv(sandwich).view(-1, 240)

		# left_context, left_weights = self.left_attn(center , left)
		# right_context, right_weights = self.right_attn(center , right)



		# concat and return prediction

		# center = self.mlp_center(center)
		# center = center.view(-1, center.size(2))
		# right_context = right_context.view(-1, right_context.size(2))
		# left_context = left_context.view(-1, left_context.size(2))
		# print("center", center.size())
		# exit()
		# left = left_attn_value
		# right = right_attn_value

		# print(center.size(),left.size(),right.size())
		# predictions = torch.cat([conv_out, center], 1)
		# appending stuff like this.
		predictions = torch.cat([note_pred_ln, note_pred_rn], 1)
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
		# pad center
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
			#  keeping the embedding size of center to be same.
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

			# print("notes" , notes)
			# print("metas" , metas)




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
		# print("no of notes" , self.num_notes_per_voice[0])
		# hot_encode_chorale = torch.eye(self.num_notes_per_voice[0])[tensor_chorale]
		# print("hot encode" , hot_encode_chorale)

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
		return (left_notes, central_notes, right_notes), label #, hot_encode_chorale

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

class Attention(nn.Module):
	""" Applies attention mechanism on the `context` using the `query`.

	**Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
	their `License
	<https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

	Args:
		dimensions (int): Dimensionality of the query and context.
		attention_type (str, optional): How to compute the attention score:

			* dot: :math:`score(H_j,q) = H_j^T q`
			* general: :math:`score(H_j, q) = H_j^T W_a q`

	Example:

		 >>> attention = Attention(256)
		 >>> query = torch.randn(5, 1, 256)
		 >>> context = torch.randn(5, 5, 256)
		 >>> output, weights = attention(query, context)
		 >>> output.size()
		 torch.Size([5, 1, 256])
		 >>> weights.size()
		 torch.Size([5, 1, 5])
	"""

	def __init__(self, dimensions, attention_type='general'):
		super(Attention, self).__init__()

		if attention_type not in ['dot', 'general']:
			raise ValueError('Invalid attention type selected.')

		self.attention_type = attention_type
		if self.attention_type == 'general':
			self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

		self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
		self.softmax = nn.Softmax(dim=-1)
		self.tanh = nn.Tanh()

	def forward(self, query, context):
		"""
		Args:
			query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
				queries to query the context.
			context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
				overwhich to apply the attention mechanism.

		Returns:
			:class:`tuple` with `output` and `weights`:
			* **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
			  Tensor containing the attended features.
			* **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
			  Tensor containing attention weights.
		"""
		batch_size, output_len, dimensions = query.size()
		query_len = context.size(1)

		if self.attention_type == "general":
			query = query.view(batch_size * output_len, dimensions)
			query = self.linear_in(query)
			query = query.view(batch_size, output_len, dimensions)

		# TODO: Include mask on PADDING_INDEX?

		# (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
		# (batch_size, output_len, query_len)
		attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

		# Compute weights across every context sequence
		attention_scores = attention_scores.view(batch_size * output_len, query_len)
		attention_weights = self.softmax(attention_scores)
		attention_weights = attention_weights.view(batch_size, output_len, query_len)

		# (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
		# (batch_size, output_len, dimensions)
		mix = torch.bmm(attention_weights, context)

		# concat -> (batch_size * output_len, 2*dimensions)
		combined = torch.cat((mix, query), dim=2)
		combined = combined.view(batch_size * output_len, 2 * dimensions)

		# Apply linear_out on every 2nd dimension of concat
		# output -> (batch_size, output_len, dimensions)
		output = self.linear_out(combined).view(batch_size, output_len, dimensions)
		output = self.tanh(output)

		return output, attention_weights


# def rle(notes):
# 	t=-1
# 	durations=[]
# 	notes_list=[]
# 	for i in notes:
# 		if i==t:
# 			durations[-1]+=1
# 		else :
# 			durations.append(1)
# 			notes_list.append(i)
# 			t=i
# 	return durations,notes_list	


# def tensor_to_rle(tensor_score):
#         """
#         :param tensor_score: (num_voices, length)
#         :return: music21 score object
#         """
#         slur_indexes = [note2index[SLUR_SYMBOL]
#                         for note2index in self.note2index_dicts]

#         score = []
#         # score = music21.stream.Score()
#         for voice_index, voice in enumerate(
#                 tensor_score):
#             # part = stream.Part(id='part' + str(voice_index))
#             dur = 0
#             f = music21.note.Rest()
#             for note_index in [n.item() for n in voice]:
#                 # if it is a played note
#                 if not note_index == slur_indexes[voice_index]:
#                     # add previous note
#                 if dur > 0:
#                     f.duration = music21.duration.Duration(dur / self.subdivision)
#                     part.append(f)

#                 dur = 1
#                 f = standard_note(index2note[note_index])
#                 else:
#                     dur += 1
#             # add last note
#             # f.duration = music21.duration.Duration(dur / self.subdivision)
#             # part.append(f)
#             score.append(dur/self.subdivision)
#         return score
