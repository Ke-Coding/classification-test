from collections import OrderedDict
import torch as tc
import torch.nn as nn
import torchsummary
import os


__all__ = ['FCNet']


class FCNet(nn.Module):
	def __init__(self, input_dim, output_dim,
				 hid_dims=[72, 32, 16],
				 dropout_p=None
				 ):
		r"""
		:param input_dim(int): input dimention
		:param output_dim(int): output dimention
		:param hid_dims(List[int]): hidden layers' dimentions
			- len(hid_dims): the number of the hidden layers
		"""
		super(FCNet, self).__init__()
		self.input_dim, self.output_dim = input_dim, output_dim
		self.hid_dims = hid_dims
		self.using_dropout = dropout_p is not None

		self.bn0 = nn.BatchNorm1d(input_dim)	# this layer has a huge influence on acc
		self.backbone = self._make_backbone(
			len(hid_dims),
			[input_dim] + hid_dims[:-1],
			hid_dims
		)
		if self.using_dropout:
			self.dropout_p = dropout_p
			self.dropout = nn.Dropout(p=dropout_p)
		self.classifier = nn.Linear(hid_dims[-1], output_dim, bias=True)
		self._init_params()
	
	def forward(self, x):
		"""
		:param x(3D Tensor, C*W*H): input(images)
		input(images) ==[view]==> flatten inputs ==[batchnorm0]==> normalized flatten inputs
		==[backbone]==> features ==[dropout]==> sparse features ==[classifier]==> output(logits)
		"""
		flatten_x = x.view(x.size(0), -1)	# x.size(0): batch size(the number of images in each mini-batch)
		flatten_x = self.bn0(flatten_x)
		features = self.backbone(flatten_x)
		if self.using_dropout:
			features = self.dropout(features)
		logits = self.classifier(features)
		return logits
	
	def _make_backbone(self, num_layers, in_dims, out_dims):
		backbone = OrderedDict()
		name_prefix = 'linear'
		for i in range(num_layers):
			name = name_prefix + "_%d" % i
			backbone[name] = nn.Sequential(
				nn.Linear(in_dims[i], out_dims[i], bias=False),
				nn.BatchNorm1d(out_dims[i]),
				nn.ReLU(inplace=True),
			)
		return nn.Sequential(backbone)
	
	def _init_params(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, std=0.001)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
	
	def save(self, final_acc_str):
		save_path = 'saves\\fc_saves\\' + final_acc_str
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		file_name = 'hid_dims[%s]_%s.pkl' % ('-'.join([str(e) for e in self.hid_dims]),
											 'dropout%g' % self.dropout_p if self.using_dropout else 'no_dropout')
		file_path = os.path.join(save_path, file_name)
		tc.save(self.state_dict(), file_path)
		print('model saved successfully at %s' % file_path)
	
	def load(self, final_acc_str):
		save_path = 'saves\\fc_saves\\' + final_acc_str
		assert os.path.exists(save_path)
		file_name = 'hid_dims[%s]_%s.pkl' % ('-'.join([str(e) for e in self.hid_dims]),
											 'dropout%g' % self.dropout_p if self.using_dropout else 'no_dropout')
		file_path = os.path.join(save_path, file_name)
		self.load_state_dict(tc.load(file_path))
		print('model loaded successfully at %s' % file_path)


if __name__ == '__main__':  # testing
	net: FCNet = FCNet(input_dim=1*28*28, output_dim=10, dropout_p=0.2)
	torchsummary.summary(net, (1, 28, 28))
	net(tc.rand((2, 1, 28, 28), dtype=tc.float32))
