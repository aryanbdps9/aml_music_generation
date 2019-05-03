import torch
import torch.nn as nn

class self_attn_mini(nn.Module):
	def __init__(self, input_size):
		super(self_attn_mini, self).__init__()
		self.input_size = input_size
		self.key = nn.Linear(input_size, input_size, bias=False)
		self.query = nn.Linear(input_size, input_size, bias=False)
		self.value = nn.Linear(input_size, input_size, bias=False)
		self.softmax = nn.Softmax(dim=2)
		self.activate_cuda = torch.cuda.is_available()
	def forward(self, input, pos_enc=None):
		q = self.query(input)
		k = self.key(input)
		v = self.value(input)
		if pos_enc is not None:
			wt_mat = q @ torch.transpose(k+pos_enc, 1, 2)
		else:
			wt_mat = q @ torch.transpose(k,1,2)
		wt_mat = wt_mat / (self.input_size ** 0.5)
		wt_mat_softmaxed = self.softmax(wt_mat)
		transformed = wt_mat_softmaxed @ v
		return transformed

class self_attn_monster(nn.Module):
	def __init__(self, input_size, nlayers):
		super(self_attn_monster, self).__init__()
		assert(nlayers > 0)
		self.feed_fwds = nn.ModuleList()
		for _ in range(nlayers):
			self.feed_fwds.append(nn.Linear(input_size, input_size))
		self.activation = nn.LeakyReLU()
		self.sam = self_attn_mini(input_size)
		self.input_size = input_size
		self.pos_embedding = nn.Embedding(63, input_size)
		self.activate_cuda = torch.cuda.is_available()
	def forward(self, input, lr='l'):
		bs = input.size(0)
		ntsteps = input.size(1)
		if lr == 'l':
			indices = torch.arange(ntsteps) + 31 - ntsteps
			if self.activate_cuda:
				indices = indices.cuda()
		else:
			indices = torch.arange(ntsteps) + 32
			if self.activate_cuda:
				indices = indices.cuda()

		positional_emb = self.pos_embedding(indices).unsqueeze(0).repeat(bs, 1,1)
		pos_inp = input# + positional_emb
		for li, layer in enumerate(self.feed_fwds):
			pos_inp = self.sam(pos_inp, positional_emb)
			pos_inp = layer(pos_inp)
			pos_inp = self.activation(pos_inp)
		return pos_inp

