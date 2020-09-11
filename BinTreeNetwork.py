import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from math import sqrt

def init_weights_unif(in_dim,out_dim,prevent_wild=True,preserve=True,alternating=False,**kwargs):
	x = torch.rand((out_dim,in_dim))					#Uniform [0,1]
	x = x*2-1												#Uniform [-1,1]
	if prevent_wild:
		x = x/sqrt(out_dim)								#Prevents early wild behaviour
	if preserve:
		eye = torch.eye(out_dim,in_dim)
		if alternating:
			eye = eye*(-1)**torch.arange(in_dim)	#Improves negative memory (?)
		x = x + torch.eye(out_dim,in_dim)			#Makes memory function easier
	return(x)

def init_bias_unif(dim,prevent_wild=True,**kwargs):
	x = torch.rand((dim,))							#Uniform [0,1]
	x = x*2-1											#Uniform [-1,1]
	if prevent_wild:
		x = x/sqrt(dim)								#Prevents early wild behaviour
	return(x)

class ComponentNetwork(nn.Module):
	def __init__(self,in_dim=2,out_dim=1,width=2,depth=1,**kwargs):
		super(ComponentNetwork,self).__init__()
		self.in_layer = nn.Parameter(init_weights_unif(in_dim,width,**kwargs))
		self.in_bias = nn.Parameter(init_bias_unif(width,**kwargs))
		depth = depth - 1
		self.hidden_layer = nn.ParameterList([
			nn.Parameter(init_weights_unif(width,width,**kwargs)) for _ in range(depth)
		])
		self.hidden_bias = nn.ParameterList([
			nn.Parameter(init_bias_unif(width,**kwargs)) for _ in range(depth)
		])
		self.out_layer = nn.Parameter(init_weights_unif(width,out_dim,**kwargs))
	def forward(self,x):
		out = F.relu(F.linear(x,self.in_layer,self.in_bias))
		for i in range(len(self.hidden_layer)):
			hl = self.hidden_layer[i]
			hb = self.hidden_bias[i]
			out = F.relu(F.linear(out,hl,hb))
		out = F.linear(out,self.out_layer)
		return(out)



class BinTreeNetwork(nn.Module):
	def __init__(self,in_dim,out_dim,comp_width,comp_depth,tree_width,tree_depth,**kwargs):
		super(BinTreeNetwork,self).__init__()
		self.component = ComponentNetwork(2,1,comp_width,comp_depth,**kwargs)
		self.in_left_layer = nn.Parameter(init_weights_unif(in_dim,tree_width,**kwargs))
		self.in_right_layer = nn.Parameter(init_weights_unif(in_dim,tree_width,**kwargs))
		self.in_left_bias = nn.Parameter(init_bias_unif(tree_width,**kwargs))
		self.in_right_bias = nn.Parameter(init_bias_unif(tree_width,**kwargs))
		tree_depth = tree_depth-1
		self.tree_left_layer = nn.ParameterList([
			nn.Parameter(init_weights_unif(tree_width,tree_width,**kwargs)) for _ in range(tree_depth)
		])
		self.tree_right_layer = nn.ParameterList([
			nn.Parameter(init_weights_unif(tree_width,tree_width,**kwargs)) for _ in range(tree_depth)
		])
		self.tree_left_bias = nn.ParameterList([
			nn.Parameter(init_bias_unif(tree_width,**kwargs)) for _ in range(tree_depth)
		])
		self.tree_right_bias = nn.ParameterList([
			nn.Parameter(init_bias_unif(tree_width,**kwargs)) for _ in range(tree_depth)
		])
		self.out_layer = nn.ParameterList([ 
			nn.Parameter(init_weights_unif(in_dim,out_dim,**kwargs))
		]+[
			nn.Parameter(init_weights_unif(tree_width,out_dim,**kwargs)) for _ in range(tree_depth)
		])
		self.out_bias = nn.Parameter(init_bias_unif(out_dim,**kwargs))
		self.tree_depth = tree_depth
		self.tree_width = tree_width
	def forward(self,x):
		ll = self.in_left_layer
		rl = self.in_right_layer
		lb = self.in_left_bias
		rb = self.in_right_bias
		ol = self.out_layer[0]
		left_input = F.linear(x,ll,lb)
		right_input = F.linear(x,rl,rb)
		out = F.linear(x,ol)
		for i in range(self.tree_depth):
			res = []
			for j in range(self.tree_width):
				li_j = left_input[j]
				ri_j = right_input[j]
				comp_input = torch.tensor([li_j,ri_j],dtype=li_j.dtype,requires_grad=True)
				res.append(comp_input)
			res = torch.tensor(res,requires_grad=True,dtype=out.dtype)
			ll = self.tree_left_layer[i]
			rl = self.tree_right_layer[i]
			lb = self.tree_left_bias[i]
			rb = self.tree_right_bias[i]
			ol = self.out_layer[i+1]
			left_input = F.linear(res,ll,lb)
			right_input = F.linear(res,rl,rb)
			out = out + F.linear(res,ol)
		out = out + self.out_bias
		return(out)

net = BinTreeNetwork(4,1,3,3,3,3)
x = torch.randn((4,)).float()
print(net.forward(x))


N=10**5
d=2
X = torch.rand((N,d))*2-1	#Unif[-1,1]^d sample of size N
Y = torch.max(X,axis=1)[0]	#Source regression function
Y = Y.view(-1,1)				#Ensure (N,1) dimensional shape
loss = nn.MSELoss()			#create Mean Squared Error loss object (behaves as a function)
gamma = .01
Y_noisy = Y + gamma*torch.randn_like(Y)	#Normal noise sample with mu=0, std=gamma
drop_out = nn.Dropout(p=1-1/sqrt(2))

def test(net):
	with torch.no_grad():	#Disable gradient tracker
		#Create a test list of dim (Nt, d)
		ax = torch.linspace(-1,1,10**2)
		Xt1, Xt2 = torch.meshgrid((ax,ax))
		Xt1 = Xt1.reshape(-1,1)
		Xt2 = Xt2.reshape(-1,1)
		Xt = torch.cat((Xt1,Xt2),axis=1)
		#Create target list of dim (Nt, 1)
		Yt = torch.max(Xt,axis=1)[0]
		Yt = Yt.view(-1,1)
		#Compute predictions
		Y_predt = net(Xt.float())
		#Compute the difference
		diff = torch.abs(Yt-Y_predt)
		#Compute the various norms
		sup_norm = torch.max(diff)
		print('Sup norm: {0}'.format(sup_norm))
		taxi_cab = torch.sum(diff)
		print('L1 norm: {0}'.format(taxi_cab))
		euclid = torch.sqrt(torch.sum(diff**2))
		print('L2 norm: {0}'.format(euclid))


for w in range(1,34):
	width=w*3
	print('Width: {0}'.format(width))
	net = Network(in_dim=X.shape[1],out_dim=Y.shape[1],width=width)
	#net.clamp(8/sqrt(width))
	optimizer = optim.SGD(net.parameters(),lr=0.001)	#Initialize Adam optimizer with default parameters
	
	#train
	for i in range(N):
		# Collect data point
		x = X[i].view(1,-1)
		y = Y_noisy[i].view(1,-1)
		# Clear gradient
		optimizer.zero_grad()
		# Calculate prediction
		y_pred = net(x.float())
		# Compute loss
		l = loss(y,y_pred)
		l.backward()
		# Optimization step
		optimizer.step()
		#net.clamp(8/sqrt(width))
	 
	#test
	test(net)
