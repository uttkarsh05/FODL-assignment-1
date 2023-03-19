import numpy as np
import matplotlib.pyplot as plt
import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.keras.datasets import fashion_mnist,mnist
from torch.utils.data import dataloader
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
import tensorflow as tf
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import wandb
import argparse

class SimpleNamespace:
    def __init__(self, /, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        items = (f"{k}={v!r}" for k, v in self.__dict__.items())
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        if isinstance(self, SimpleNamespace) and isinstance(other, SimpleNamespace):
           return self.__dict__ == other.__dict__
        return NotImplemented

default_config = SimpleNamespace(
	wandb_project = 'sweeps',
	wandb_entity = 'uttakarsh05',
	dataset = 'fashion_mnist',
	epochs = 10,
	batch_size = 8,
	loss = 'cross_entropy',
	optimizer = 'adam',
	learning_rate = 0.000704160852345564,
	momentum = 0.9,
	beta = 0.9,
	beta1 = 0.9,
	beta2 = 0.99,
	epsilon = 1e-10,
	weight_decay = 0,
	weight_initialization = 'He',
	num_layers = 2,
	hidden_size = 128,
	activation = 'relu',
	keep_prob = 1.0,
)

def parse_args():
	argparser = argparse.ArgumentParser(description = 'Processing Hyperparameters')
	argparser.add_argument('-wp','--wandb_project',type = str,default = default_config.wandb_project,help = 'wandb project name')
	argparser.add_argument('-we','--wandb_entity',type = str,default = default_config.wandb_entity,help = 'wandb username/entity name')
	argparser.add_argument('-d','--dataset',type = str,default = default_config.dataset,help = 'dataset name')
	argparser.add_argument('-e','--epochs',type = int,default = default_config.epochs,help = 'no of epochs')
	argparser.add_argument('-b','--batch_size',type = int,default = default_config.batch_size,help = 'batch size')
	argparser.add_argument('-l','--loss',type = str,default = default_config.loss,help = 'loss function name')
	argparser.add_argument('-o','--optimizer',type = str,default = default_config.optimizer,help = 'optimizer name')
	argparser.add_argument('-lr','--learning_rate',type = float,default = default_config.learning_rate,help = 'learning rate')
	argparser.add_argument('-m','--momentum',type = float,default = default_config.momentum,help = 'beta value used for momentum optimizer')
	argparser.add_argument('-beta','--beta',type = float,default = default_config.beta,help = 'beta value used for rmsprop')
	argparser.add_argument('-beta1','--beta1',type = float,default = default_config.beta1,help = 'beta1 used by adam and nadam')
	argparser.add_argument('-beta2','--beta2',type = float,default = default_config.beta2,help = 'beta2 used by adam and nadam')
	argparser.add_argument('-eps','--epsilon',type = float,default = default_config.epsilon,help = 'epsilon value used by optimizers')
	argparser.add_argument('-w_d','--weight_decay',type = float,default = default_config.weight_decay,help = 'weight decay (lamda) value for l2 regularization')
	argparser.add_argument('-nhl','--num_layers',type = int,default = default_config.num_layers,help = 'number of hidden layers')
	argparser.add_argument('-sz','--hidden_size',type = int,default = default_config.hidden_size,help = 'size of every hidden layer')
	argparser.add_argument('-a','--activation',type = str,default = default_config.activation,help = 'activation name')
	argparser.add_argument('-kp','--keep_prob',type = float,default = default_config.keep_prob,help = 'probability of a neuron to be dropped')
	argparser.add_argument('-w_i','--weight_init',type = str,default = default_config.weight_initialization,help = 'activation name')

	args = argparser.parse_args()
	vars(default_config).update(vars(args))
	return 

def transform_data(X,y):
	X_train = X.reshape(X.shape[0],-1)
	y_train = np.zeros((y.size,y.max()+1))
	y_train[np.arange(y.size),y]=1
	return X_train,y_train  

def get_data(dataset):
	
	if dataset=='mnist':
		(train_data,train_labels),(test_data,test_labels) = mnist.load_data()

	elif dataset =='fashion_mnist':
		(train_data,train_labels),(test_data,test_labels) = fashion_mnist.load_data()


	X_train,y_train = transform_data(train_data,train_labels)

	X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size= 0.1)

	X_test,y_test = transform_data(test_data,test_labels)

	sc = StandardScaler()
	sc.fit(X_train)

	X_train = sc.transform(X_train)
	X_val = sc.transform(X_val)
	X_test = sc.transform(X_test)


	return [(X_train,y_train),(X_val,y_val),(X_test,y_test)]


def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
	return sigmoid(x)*(1-sigmoid(x))

# define vectorized sigmoid
sigmoid_f = np.vectorize(sigmoid)
sigmoid_grad_f = np.vectorize(sigmoid_grad)


def tanh(x):
	return np.tanh(x)

def tanh_grad(x):
	return (1-np.square(np.tanh(x)))

# define vectorized tanh
tanh_f = np.vectorize(tanh)
tanh_grad_f = np.vectorize(tanh_grad)

def Relu(x):
	try:
		return max(x,0)
	except:
		return np.inf

		
def Relu_grad(x):
	try:
		if x>0:
			return 1
		else:
			return 0
	except:
		return np.inf
	
# define vectorized Relu
Relu_f = np.vectorize(Relu)
Relu_grad_f = np.vectorize(Relu_grad)


class feedforward_NN():
	def __init__(self,input_size=784,output_size=10):
		self.input_size = input_size
		self.output_size = output_size
		self.keep_prob = 1
		self.hidden_layers = None
		self.hidden_size = None
		self.activation = None
		self.optimizer = None
		self.weight_intializer = None
		self.lamda = None
		self.epochs = None
		self.lr = None
		self.batch_size = None
		self.loss = None
		self.layers = [input_size]
		self.mask = []
		self.w_mask = []
		self.master_weights = [[0,0]]
		self.weights = []
		self.bias = []
		self.layer_outputs = []
		self.e = []
		self.d = []
		self.gprime = []
		self.del_w = []
		self.del_b = []
		self.cost_fun = []
		self.val_cost_fun = []
		self.accuracy = []
		self.val_accuracy = []
		self.last_update = []
	
	def softmax(self, z):
		exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
		return exp_z / np.sum(exp_z, axis=0, keepdims=True)
	
	def forward_prop(self,X):
		
		a = X.T
		a = np.multiply(a,self.mask[0])
		self.layer_outputs = []
		self.g_prime = []
		
		for i in range(len(self.weights)-1):
			
			W = np.multiply(self.weights[i],self.w_mask[i])
			b = np.multiply(self.bias[i],self.mask[i+1])
			z = W@a + b
			#print('layer = ',i+1)
			#print('max w = ',np.round(np.max(self.weights[i]),4))
			#print('max z = ' ,np.round(np.max(z),4))
			a = self._activation(z)
			a = np.multiply(a,self.mask[i+1])/self.keep_prob
			self.layer_outputs.append([z,a])
			
			#g_prime = self._activation_grad(z)
			#self.gprime.append(g_prime)
			
		W = np.multiply(self.weights[-1],self.w_mask[-1])
		b = self.bias[-1]
		zl = W@a + b
		o = self.softmax(zl)
		self.layer_outputs.append([zl,o])
		
		return o
	
	def _forward_prop(self,X):
		# doesn't updates the layer outputs and gprimes
		a = X.T
		for i in range(len(self.weights)-1):
			#print(a.shape,self.weights[i].shape,self.bias[i].shape)
			z = self.weights[i]@a+self.bias[i]
			a = self._activation(z)
			#self.layer_outputs.append([z,a])
			
			#g_prime = self._activation_grad(z)
			#self.gprime.append(g_prime)
			
		zl = self.weights[-1]@a+self.bias[-1]
		o = self.softmax(zl)
		#self.layer_outputs.append([zl,o])
		
		return o
	
	def _find_grad_loss(self,y,y_hat):
		if self.loss =="cross_entropy":
			epsilon = 1e-8
			e = -y/(y_hat+epsilon)
			d = -(y-y_hat)

		elif self.loss == 'mean_squared_error':
			e = -(y-y_hat)
			sum_matrix = np.sum(np.multiply(y,e),axis = 0)
			d = np.multiply(y,sum_matrix) + np.multiply(y_hat,e)
			#d = 2*d
			#sum_matrix = np.ones(y_hat.shape)-y_hat
			#d = np.multiply(e,np.multiply(y_hat,sum_matrix))
			#d = 2*d

		return e,d

	

	def back_prop(self,X,y):
		y = y.T
		y_hat = self.layer_outputs[-1][1]
		 
		self.e = []
		self.d = []
		self.del_w = []
		self.del_b = []
		
		#e = -y/(y_hat+1e-8)
		#d = -(y-y_hat)

		e,d = self._find_grad_loss(y,y_hat)
		
		self.e.append(e)
		self.d.append(d)
		
		for i in range(self.hidden_layers-1,-1,-1):
			
			a_l_minus_one = self.layer_outputs[i][1]
			
			del_w = ((self.d[-1] @ a_l_minus_one.T)+ self.lamda*self.weights[i+1])/(self.d[-1].shape[1])
			self.del_w.append(del_w)
			
			del_b = np.mean(self.d[-1],axis=1).reshape(self.d[-1].shape[0],1)+self.lamda*self.bias[i+1]/(self.d[-1].shape[1])
			self.del_b.append(del_b)
			
			WT = np.multiply(self.weights[i+1].T,self.w_mask[i+1].T)
			e = WT @ self.d[-1]
			e = np.multiply(e,self.mask[i+1])
			self.e.append(e)
			
			z = self.layer_outputs[i][0]
			g_prime = self._activation_grad(z)
			#g_prime = self.gprime[i]
			d = np.multiply(g_prime,self.e[-1])
			self.d.append(d)
		
		a0 = X.T
		a0 = np.multiply(a0,self.mask[0])
		
		del_w = (self.d[-1] @ a0.T)/self.d[-1].shape[1] + (self.lamda*self.weights[0])/self.d[-1].shape[1]
		self.del_w.append(del_w)
		
		del_b = np.mean(self.d[-1],axis = 1).reshape(self.d[-1].shape[0],1) + (self.lamda*self.bias[0])/(self.d[-1].shape[1])
		self.del_b.append(del_b)
		
		self.del_w.reverse()
		self.del_b.reverse()
		

		
	
	def _get_loss(self,y,y_hat):
		epsilon = 1e-8
		if self.loss == 'cross_entropy':
			loss = np.multiply(np.log(y_hat+epsilon),y)
			loss = -np.sum(loss)

		elif self.loss == 'mean_squared_error':
			loss = np.square(y-y_hat)
			loss = np.sum(loss)

		return loss




	def cost(self,X,y):
		y_true = y.T
		m = y_true.shape[1]
		#print('no of examples = ',m)
		y_hat = self._forward_prop(X)
		
		#loss = np.multiply(np.log(y_hat+epsilon),y_true)
		loss = self._get_loss(y_true,y_hat)
		l2norm = sum([np.sum(np.square(self.weights[i])) for i in range(len(self.weights))])+sum([np.sum(np.square(self.bias[i])) for i in range(len(self.bias))])
		om_theta = self.lamda*l2norm
		cost = loss/(2*m) + om_theta/(2*m)
		
		return cost
	
	
	def make_mask(self):
		
		mask = []
		w_mask = []
		
		p = self.keep_prob
		
		for i in range(len(self.layers)-1):
			l = self.layers[i]
			m = np.linspace(start = 0,stop=1,num = l)
			np.random.shuffle(m)
			m = np.ones(l)*(m<p)
			m = m.reshape(-1,1)
			mask.append(m)
		
		for i in range(len(mask)-1):
			w = mask[i+1]@mask[i].T
			w_mask.append(w)
			
		w = np.ones(self.layers[-1]).reshape(-1,1) @ mask[-1].T
		w_mask.append(w)
		
		self.mask = mask
		self.w_mask = w_mask
		
		
	
	def val_accuracy(self):
		try:
			return self.val_accuracy[-1]
		except:
			return 0
		   
	def predict(self,X):
		
		y_hat = self._forward_prop(X)
		y_hat = np.argmax(y_hat.T,axis = 1)
		
		return y_hat
				   
	def print_report(self):
		print('------------------------')
		print('Model Reports')
		print('Weight initializer = ',self.weight_intializer)
		print('Loss function used = ',self.loss)
		print('Optimizer = ',self.optimizer)
		print('Layer activation = ',self.activation)
		print('Learning rate = ',self.lr)
		print('Dropout keep probability = ',self.keep_prob)
		print('Weight decay = ',self.lamda)
		print('Batch size = ',self.batch_size)
		for i in range(1,len(self.weights)+1):
			print('layer = ',i,' | ', 'weights trained = ',self.weights[i-1].shape )
		try:
			print('Final validation accuracy = ',np.round(self.val_accuracy[-1],4),' Final training accuracy = ',np.round(self.accuracy[-1],4))
		except:
			print('empty')
		print('Total no of epochs trained = ',self.epochs)

		
	def _activation(self,z):
		
		if self.activation =='sigmoid':
			
			a = sigmoid_f(z)
			
		elif self.activation =='tanh':
			
			a = tanh_f(z)
		
		elif self.activation =='relu':
			
			a = Relu_f(z)
		
		return a

	
	def _activation_grad(self,z):
		
		if self.activation =='sigmoid':
			
			g_prime = sigmoid_grad_f(z)
			
		if self.activation =='tanh':
			
			g_prime = tanh_grad_f(z)
			
		if self.activation =='relu':
			
			g_prime = Relu_grad_f(z)
		
		return g_prime
			
	
	def _initialize_wandb(self,initializer):
		
		self.weights = []
		self.bias = []
		layers = self.layers
		for i in range(1,len(layers)):
			if initializer=='Xavier':
				#np.random.seed(42)
				w = np.random.normal(0,1,size = (layers[i],layers[i-1]))*np.sqrt(1/layers[i-1])
			elif initializer=='random':
				w = np.random.normal(0,1,size = (layers[i],layers[i-1]))
			
			elif initializer=='He':
				w = np.random.normal(0,1,size=(layers[i],layers[i-1]))*np.sqrt(2/(layers[i-1]))
				
			if self.activation=='relu':
				b = 0.01*np.ones((layers[i],1))
			else:
				b = np.random.normal(0,1,size=(layers[i],1))
			
			self.weights.append(w)
			self.bias.append(b)
		#self.weights = w
		#self.bias = b

def get_optimizer(model,optimizer,learning_rate):
		
	if optimizer == 'sgd':
		optim = SGD(model,learning_rate)

	if optimizer == 'momentum':
		#print('it came here')
		optim = Momentum(model,learning_rate)

	if optimizer == 'nag':
		optim = NAG(model,learning_rate)

	if optimizer == 'rmsprop':
		optim = RMSProp(model,learning_rate)

	if optimizer == 'adam':
		optim = Adam(model,learning_rate)

	if optimizer == 'nadam':
		optim = NAdam(model,learning_rate)

	return optim

class SGD():
	
	def __init__(self,model,learning_rate):
		self.model = model
		self.learning_rate = learning_rate
	
	def update(self,X_train,y_train):
		
		y_hat = self.model.forward_prop(X_train)
		
		self.model.back_prop(X_train,y_train)
		#print(self.model.del_b[0].shape)
		
		for i in range(len(self.model.weights)):
		
			self.model.weights[i] = self.model.weights[i] - self.learning_rate*self.model.del_w[i]
			
			self.model.bias[i] = self.model.bias[i] - self.learning_rate*self.model.del_b[i]
		
		
class Momentum():
	
	def __init__(self,model,learning_rate,beta=0.9):
		
		self.model = model
		
		self.learning_rate = learning_rate
		self.beta = beta
		
		self.ut = []
		self.bt = []
		
		for i in range(len(self.model.weights)):
			
			ut = np.zeros(self.model.weights[i].shape)
			self.ut.append(ut)
			
			bt = np.zeros(self.model.bias[i].shape)
			self.bt.append(bt)
			
	
	def update(self,X_train,y_train):
		#print(self.ut[0][0][0])
		#Propogating forward and collecting output of every layer and gradients of activations
		#print('came here')
		y_hat = self.model.forward_prop(X_train)
		
		#Back propogating and collecting gradient for every weight matrix 
		self.model.back_prop(X_train,y_train)
		
		for i in range(len(self.model.weights)):
			#print('came here also')
			#print('max del_w ',np.round(np.max(self.model.del_w[i]),4))
			ut = self.beta*self.ut[i] + self.model.del_w[i]
			self.model.weights[i] = self.model.weights[i] - self.learning_rate*ut
			#if i==0:
				#print('layer = ',i,' norm = ',np.linalg.norm(self.ut[i]-ut))
			self.ut[i] = ut
			
			bt = self.beta*self.bt[i] + self.model.del_b[i]
			self.model.bias[i] = self.model.bias[i] - self.learning_rate*bt
			self.bt[i] = bt
		

class NAG():
	
	def __init__(self,model,learning_rate,beta=0.9):
		self.model = model
		
		self.learning_rate = learning_rate
		self.beta = beta
		
		self.ut = []
		self.bt = []
		
		for i in range(len(self.model.weights)):
			
			ut = np.zeros(self.model.weights[i].shape)
			self.ut.append(ut)
			
			bt = np.zeros(self.model.bias[i].shape)
			self.bt.append(bt)
	
	def update(self,X_train,y_train):
		
		y_hat = self.model.forward_prop(X_train)
		weights = []
		bias = []
		for i in range(len(self.model.weights)):
			weights.append(self.model.weights[i])
			self.model.weights[i] = self.model.weights[i] - self.beta*self.ut[i]
			
			bias.append(self.model.bias[i])
			self.model.bias[i] = self.model.bias[i] - self.beta*self.bt[i]
		
		#y_hat = self.model.forward_prop(X_train)
		
		self.model.back_prop(X_train,y_train)
		
		for i in range(len(self.model.weights)):
			
			ut = self.beta*self.ut[i] + self.model.del_w[i]
			#self.model.weights[i] = self.model.weights[i] + self.beta*self.ut[i]
			self.model.weights[i] = weights[i] - self.learning_rate*ut
			self.ut[i] = ut
			
			bt = self.beta*self.bt[i] + self.model.del_b[i]
			#self.model.bias[i] = self.model.bias[i] + self.beta*self.bt[i]
			self.model.bias[i] = bias[i] - self.learning_rate*bt
			self.bt[i] = bt
		
		
class RMSProp():
	
	def __init__(self,model,learning_rate,beta=0.9):
		self.model = model
		
		self.learning_rate = learning_rate
		self.beta = beta
		
		self.ut = []
		self.bt = []
		
		for i in range(len(self.model.weights)):
			
			ut = np.zeros(self.model.weights[i].shape)
			self.ut.append(ut)
			
			bt = np.zeros(self.model.bias[i].shape)
			self.bt.append(bt)
	
	def update(self,X_train,y_train):
		
		y_hat = self.model.forward_prop(X_train)
		
		epsilon = 1e-10
		#Back propogating and collecting gradient for every weight matrix 
		self.model.back_prop(X_train,y_train)
		
		for i in range(len(self.model.weights)):
			
			ut = self.beta*self.ut[i] + (1-self.beta)*np.square(self.model.del_w[i])
			self.model.weights[i] = self.model.weights[i] - (self.learning_rate/(np.sqrt(ut)+epsilon))*(self.model.del_w[i])
			self.ut[i] = ut
			
			bt = self.beta*self.bt[i] + (1-self.beta)*np.square(self.model.del_b[i])
			self.model.bias[i] = self.model.bias[i] - (self.learning_rate/(np.sqrt(bt)+epsilon))*self.model.del_b[i]
			self.bt[i] = bt
		
		
		
class Adam():
	
	def __init__(self,model,learning_rate,beta1=0.9,beta2=0.999):
		
		self.model = model
		
		self.learning_rate = learning_rate
		self.beta1 = beta1
		self.beta2 = beta2
		
		self.mt = []
		self.vt = []
		
		self.mbt = []
		self.vbt = []
		
		self.t = 1
		
		for i in range(len(self.model.weights)):
			
			mt = np.zeros(self.model.weights[i].shape)
			vt = np.zeros(self.model.weights[i].shape)
			self.mt.append(mt)
			self.vt.append(vt)
			
			mbt = np.zeros(self.model.bias[i].shape)
			vbt = np.zeros(self.model.bias[i].shape)
			self.mbt.append(mbt)
			self.vbt.append(vbt)
	
	def update(self,X_train,y_train):
		
		y_hat = self.model.forward_prop(X_train)
		
		self.model.back_prop(X_train,y_train)
		
		epsilon = 1e-10
		
		#print('updates = ',self.t)
		for i in range(len(self.model.weights)):
			
			mt = self.beta1*self.mt[i] + (1-self.beta1)*(self.model.del_w[i])
			mt_hat = mt/(1-np.power(self.beta1,self.t))
			
			vt = self.beta2*self.vt[i] + (1-self.beta2)*np.power(self.model.del_w[i],2)
			vt_hat = vt/(1-np.power(self.beta2,self.t))

			self.model.weights[i] = self.model.weights[i] - (self.learning_rate*mt_hat/(np.sqrt(vt_hat)+epsilon))
			
			self.mt[i] = mt
			self.vt[i] = vt
			
			mbt = self.beta1*self.mbt[i] + (1-self.beta1)*self.model.del_b[i]
			mbt_hat = mbt/(1-np.power(self.beta1,self.t))
			
			vbt = self.beta2*self.vbt[i] + (1-self.beta2)*np.power(self.model.del_b[i],2)
			vbt_hat = vbt/(1-np.power(self.beta2,self.t))
			
			self.model.bias[i] = self.model.bias[i] - self.learning_rate*mbt_hat/(np.sqrt(vbt_hat)+epsilon)
			
			self.mbt[i] = mbt
			self.vbt[i] = vbt
		
		self.t+=1
		
		
		
class NAdam():
	
	def __init__(self,model,learning_rate,beta1=0.9,beta2=0.999):
		
		self.model = model
		
		self.learning_rate = learning_rate
		self.beta1 = beta1
		self.beta2 = beta2
		
		self.mt = []
		self.vt = []
		
		self.mbt = []
		self.vbt = []
		
		self.t = 1
		
		for i in range(len(self.model.weights)):
			
			mt = np.zeros(self.model.weights[i].shape)
			vt = np.zeros(self.model.weights[i].shape)
			self.mt.append(mt)
			self.vt.append(vt)
			
			mbt = np.zeros(self.model.bias[i].shape)
			vbt = np.zeros(self.model.bias[i].shape)
			self.mbt.append(mbt)
			self.vbt.append(vbt)
		
	
	def update(self,X_train,y_train):
		
		y_hat = self.model.forward_prop(X_train)
		
		self.model.back_prop(X_train,y_train)
		
		epsilon = 1e-10
		
		beta1 = self.beta1
		
		beta2 = self.beta2
		#print('updates = ',self.t)
		
		for i in range(len(self.model.weights)):

			mt = beta1*self.mt[i] + (1-beta1)*self.model.del_w[i]
			mt_hat = mt/(1-np.power(beta1,self.t))
			
			#print(self.vt[i].shape,self.model.del_w[i].shape)
			vt = beta2*self.vt[i] + (1-beta2)*np.power(self.model.del_w[i],2)
			vt_hat = vt/(1-np.power(beta2,self.t))
			

			numerator = (beta1*mt_hat) + (1-beta1)*self.model.del_w[i]/(1-np.power(beta1,self.t))
			denominator = np.sqrt(vt_hat)+epsilon

			self.model.weights[i] = self.model.weights[i] - (self.learning_rate*numerator)/(denominator)

			self.mt[i] = mt
			self.vt[i] = vt

			mbt = beta1*self.mbt[i] + (1-beta1)*self.model.del_b[i]
			mbt_hat = mbt/(1-np.power(beta1,self.t))
			

			vbt = beta2*self.vbt[i] + (1-self.beta2)*np.power(self.model.del_b[i],2)
			vbt_hat = vbt/(1-np.power(beta2,self.t))

			numerator = (beta1*mbt_hat) + (1-beta1)*self.model.del_b[i]/(1-np.power(beta1,self.t))
			denominator = np.sqrt(vbt_hat)+epsilon

			self.model.bias[i] = self.model.bias[i] - (self.learning_rate*numerator)/(denominator)

			self.mbt[i] = mbt
			self.vbt[i] = vbt

		self.t+=1
		
		
def train(model,X,y,X_val,y_val,X_test,y_test,hidden_layers,hidden_size,activation,weight_initialization,batch_size,optimizer,learning_rate,lamda,epochs,keep_prob,loss):
	
	model.hidden_layers = hidden_layers
	model.hidden_size = hidden_size
	model.loss = loss
	model.keep_prob = keep_prob
	model.batch_size = batch_size
	
	#for i in hidden_size:
	 #   model.layers.append(i)
		
	for i in range(hidden_layers):
		model.layers.append(hidden_size)
		
	model.layers.append(model.output_size)
	
	model.activation = activation
	
	model.weight_intializer = weight_initialization
	
	model._initialize_wandb(weight_initialization)

	model.cost_fun = []
	model.val_cost_fun = []
	
	model.lamda = lamda
	
	model.optimizer = optimizer
	model.lr = learning_rate
	optim = get_optimizer(model,optimizer,learning_rate)
	
	try:
		#optim.t = 1
		pass
	except:
		pass
	
	
	model.epochs = 0
	
	indices = np.arange(X.shape[0])
	
	batches = X.shape[0]//model.batch_size
	
	for i in range(epochs):
		
		
		
		np.random.shuffle(indices)
				
		j = 1
		for k in range(batches):
			batch_indices = indices[k*model.batch_size :(k+1)*model.batch_size]
			
			X_train = X[batch_indices]
			y_train = y[batch_indices]
			model.make_mask()
			

			optim.update(X_train,y_train)
			
			j+=1
		
		
		
		try:
			#optim.t+=1
			pass
		except:
			pass
		
		model.epochs+=1
		
		model.master_weights.append(model.weights)
			
		cost = model.cost(X,y)
		accuracy = accuracy_score(model.predict(X),np.argmax(y.T,axis = 0))
		
		val_cost = model.cost(X_val,y_val)
		val_accuracy = accuracy_score(model.predict(X_val),np.argmax(y_val.T,axis = 0))

		#test_cost = model.cost(X_test,y_test)
		test_accuracy = accuracy_score(model.predict(X_test),np.argmax(y_test.T,axis=0))
		
		#if i!=0 and val_cost>(model.val_cost_fun[-1]+0.1):
		 #       break
				
		print('epochs = ',i+1 ,' | ', ' cost = ',np.round(cost,4),' | ',' val cost = ',np.round(val_cost,4),' | ',' train accuracy = ',np.round(accuracy,4)*100,' | ',' val accuracy = ',np.round(val_accuracy,4)*100,' | ',' test accuracy = ',np.round(test_accuracy,4)*100)
		print('------------------------------------------------------------------------------------')
		
		#if wb:
		wandb.log({'epochs':i+1 , 'train cost':np.round(cost,4),'val cost':np.round(val_cost,4),'train accuracy ':np.round(accuracy,4)*100,'val_accuracy':np.round(val_accuracy,4)*100,'test accuracy':np.round(test_accuracy,4)*100})

		
		model.cost_fun.append(cost)
		model.accuracy.append(accuracy)
		
		model.val_cost_fun.append(val_cost)
		model.val_accuracy.append(val_accuracy)
		
		
		
	model.print_report()


def train_wandb(config = default_config):

	run = wandb.init(project = config.wandb_project,entity = config.wandb_entity,config=config)


	config = wandb.config

	dataset = config.dataset
	data  = get_data(dataset)

	X_train,y_train = data[0]
	X_val,y_val = data[1]
	X_test,y_test = data[2]

	
	input_size,output_size = X_train.shape[1],y_train.shape[1]
	model = feedforward_NN(input_size,output_size)
	
	hidden_layers = config.num_layers
	hidden_size = config.hidden_size
	
	activation = config.activation
	
	weight_initialization = config.weight_initialization
	
	batch_size = config.batch_size
	
	optimizer = config.optimizer
	
	learning_rate = config.learning_rate
	
	lamda = config.weight_decay
	
	epochs = config.epochs
	
	keep_prob = config.keep_prob

	loss = config.loss

	n = 'nhl_'+str(hidden_layers)+'_sz_'+str(hidden_size)+'_act_'+str(activation)+'_w_init_'+str(weight_initialization)+'_b_size_'+str(batch_size)+'_optim_'+str(optimizer)+'_lr_'+str(learning_rate)+'_kp_'+str(keep_prob)+'_epoch_'+str(epochs)

	run.name = n

	
	train(model,X_train,y_train,X_val,y_val,X_test,y_test,hidden_layers,hidden_size ,activation,weight_initialization,batch_size,optimizer,learning_rate,lamda,epochs,keep_prob,loss)

		
		


			





if __name__ == "__main__":
	parse_args()
	train_wandb(default_config)

