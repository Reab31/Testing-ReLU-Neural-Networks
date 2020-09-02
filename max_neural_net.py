import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from math import sqrt


class Network(nn.Module):
	def __init__(self,in_dim=2,out_dim=1,width=3):
		super(Network,self).__init__()
		self.in_layer = nn.Parameter(torch.randn(size=(in_dim,width)))
		self.out_layer = nn.Parameter(torch.randn(size=(width,out_dim)))
	def forward(self,x,drop_out=None):
		if drop_out is None:
			out = F.relu(F.linear(x,self.in_layer.T))
			return(F.linear(out,self.out_layer.T))
		else:
			out = F.relu(F.linear(x,drop_out(self.in_layer).T))
			return(F.linear(out,drop_out(self.out_layer).T))
	def clamp(self,val):
		with torch.no_grad():
			torch.clamp(self.in_layer,-val,val,out=self.in_layer)
			torch.clamp(self.out_layer,-val,val,out=self.out_layer)

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
