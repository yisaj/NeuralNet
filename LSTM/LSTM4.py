from __future__ import division
import numpy as np
import random
import time

alpha_cypher = [' ','1','2','3','4','5','6','7','8','9','0','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','.',',','\'',':',';','-','(',')','!','?']

def alpha_encode_bible(string):
	x = []
	y = []
	for i in xrange(len(string) - 1):
		x.append(alpha_encode(string[i]))
		y.append(alpha_encode(string[i + 1]))
	return [{'x': x, 'y': y}]

def alpha_encode_data(data):
	result = []
	for datum in data:
		set = {'x': [], 'y': []}
		for x in datum[0]:
			a = [0] * 73
			a[alpha_cypher.index(x)] = 1
			set['x'].append(np.array(a)[np.newaxis].T)
		for y in datum[1]:
			b = [0] * 73
			b[alpha_cypher.index(y)] = 1
			set['y'].append(np.array(b)[np.newaxis].T)
		result.append(set)
	return result
	
def alpha_decode(binary):
	return alpha_cypher[binary.tolist().index(max(binary).tolist())]

def alpha_encode(alpha):
	result = [0] * 73
	result[alpha_cypher.index(alpha)] = 1
	return np.array(result)[np.newaxis].T
	
def tanh_one(x):
	return 1 / (1 + np.exp(-x))
	
def tanh_two(x):
	return 4 / (1 + np.exp(-x)) - 2
	
def tanh(x):
	return 2 / (1 + np.exp(-x)) - 1
	
def tanh_one_prime(x):
	return logistic(x) * logistic(-x)
	
def tanh_two_prime(x):
	return 4 * logistic(x) * logistic(-x)
	
def tanh_prime(x):
	return 2 * logistic(x) * logistic(-x)

class LSTM:
	def __init__(self, layers):
		self.layers = layers
		#weights from last layer of hidden cells to output layer
		self.w_oc = np.random.randn(layers[-1], layers[-2])
		self.b_o = np.random.randn(layers[-1], 1)
		#initial cell states and biases for each gate and cell output
		self.s = [np.random.randn(y, 1) for y in layers[1:-1]]
		self.b_s = [np.random.randn(y, 1) for y in layers[1:-1]]		
		self.b_u = [np.random.randn(y, 1) for y in layers[1:-1]]		
		self.b_f = [np.random.randn(y, 1) for y in layers[1:-1]]		
		self.b_n = [np.random.randn(y, 1) for y in layers[1:-1]]		
		#weights from each hidden unit to other hidden units in their layer and their gates
		self.w_ch = [np.random.randn(y, y) for y in layers[1:-1]]
		self.w_uh = [np.random.randn(y, y) for y in layers[1:-1]]
		self.w_fh = [np.random.randn(y, y) for y in layers[1:-1]]
		self.w_nh = [np.random.randn(y, y) for y in layers[1:-1]]
		#weights from each hidden layer to the next one
		self. w_cc = [np.random.randn(y, x) for y, x in zip(layers[1:-1], layers[:-2])]
		self. w_uc = [np.random.randn(y, x) for y, x in zip(layers[1:-1], layers[:-2])]
		self. w_fc = [np.random.randn(y, x) for y, x in zip(layers[1:-1], layers[:-2])]
		self. w_nc = [np.random.randn(y, x) for y, x in zip(layers[1:-1], layers[:-2])]
		#outputs for t = -1
		self.h = [np.random.randn(y, 1) for y in layers]
		self.h_prime = [np.random.randn(y, 1) for y in layers]
		#initial partial derivatives for cell, input gate, and forget gate deltas
		self.parw_c = [np.zeros((y, x + y)) for y, x in zip(layers[1:-1], layers[:-2])]
		self.parw_n = [np.zeros((y, x + y)) for y, x in zip(layers[1:-1], layers[:-2])]
		self.parw_f = [np.zeros((y, x + y)) for y, x in zip(layers[1:-1], layers[:-2])]
		self.parb_c = [np.zeros((y, 1)) for y in layers[1:-1]]
		self.parb_n = [np.zeros((y, 1)) for y in layers[1:-1]]
		self.parb_f = [np.zeros((y, 1)) for y in layers[1:-1]]
		
	def feed(self, input):
		self.h[0] = input
		self.net_c = []
		self.net_u = []
		self.net_f = []
		self.net_n = []
		self.y_c = []
		self.y_u = []
		self.y_f = []
		self.y_n = []
		self.y_s = []
		for j in xrange(len(self.layers) - 2):
			#calculate the nets to the cell and the gates
			self.net_c.append(np.dot(self.w_cc[j], self.h[j]) + np.dot(self.w_ch[j], self.h[j + 1]))
			self.net_u.append(np.dot(self.w_uc[j], self.h[j]) + np.dot(self.w_uh[j], self.h[j + 1]))
			self.net_f.append(np.dot(self.w_fc[j], self.h[j]) + np.dot(self.w_fh[j], self.h[j + 1]))
			self.net_n.append(np.dot(self.w_nc[j], self.h[j]) + np.dot(self.w_nh[j], self.h[j + 1]))
			#calculate the output entering cell and exiting each gate
			self.y_c.append(tanh_two(self.net_c[j] + self.b_s[j]))
			self.y_u.append(tanh_one(self.net_u[j] + self.b_u[j]))
			self.y_f.append(tanh_one(self.net_f[j] + self.b_f[j]))
			self.y_n.append(tanh_one(self.net_n[j] + self.b_n[j]))
			#calculate the state change and output from the cell
			self.s[j] = self.y_c[j] * self.y_n[j] + self.y_f[j] * self.s[j]
			self.y_s.append(tanh(self.s[j]))
			#save to property h
			self.h_prime[j + 1] = self.h[j + 1]
			self.h[j + 1] = self.y_s[j] * self.y_u[j]
		#calculate the net and output of the output layer
		self.net_o = np.dot(self.w_oc, self.h[-2])
		self.y_o = tanh_one(self.net_o + self.b_o)
		self.h_prime[-1] = self.h[-1]
		self.h[-1] = self.y_o
	
	def train(self, data, epochs, eta):
		timer = True
		for i in xrange(epochs):
			for datum in data:
				for k in xrange(len(datum['x'])):
					if timer == True:
						start = time.clock()
						
					self.feed(datum['x'][k])
					
					self.err_o = tanh_one_prime(self.net_o) * (self.y_o - datum['y'][k])
					self.w_oc -= eta * np.dot(self.err_o, self.h[-2].T)
					self.b_o -= eta * self.err_o
					
					self.err_u = [tanh_one_prime(self.net_u[-1]) * tanh(self.s[-1]) * np.dot(self.w_oc.T, self.err_o)]
					for l in xrange(len(self.layers) - 3):
						self.err_u.insert(0, np.dot(self.w_uc[-l - 1].T * tanh_one_prime(self.net_u[-l - 2]) * tanh(self.y_s[-l - 2]), self.err_u[-l - 1]))
					for m in xrange(len(self.layers) - 2):
						self.w_uc[m] -= eta * np.dot(self.err_u[m], self.h[m].T)
						self.w_uh[m] -= eta * np.dot(self.err_u[m], self.h_prime[m + 1].T)
						self.b_u[m] -= eta * self.err_u[m]
					
					self.e_s = [self.y_u[-1] * tanh_prime(self.s[-1]) * np.dot(self.w_oc.T, self.err_o)]
					for n in xrange(len(self.layers) - 3):
						self.e_s.insert(0, self.y_u[-n - 2] * tanh_prime(self.s[-n - 2] * np.dot(self.w_uc[-n - 1].T, self.err_u[-n - 1])))
					for p in xrange(len(self.layers) - 2):
						self.parw_c[p] = self.parw_c[p] * self.y_f[p] + np.dot(tanh_two_prime(self.net_c[p]) * self.y_n[p], np.append(self.h[p], self.h_prime[p + 1])[np.newaxis])
						self.parw_n[p] = self.parw_n[p] * self.y_f[p] + np.dot(tanh(self.net_c[p]) * tanh_one_prime(self.net_n[p]), np.append(self.h[p], self.h_prime[p + 1])[np.newaxis])
						self.parw_f[p] = self.parw_f[p] * self.y_f[p] + np.dot(tanh(self.s[p]) * tanh_one_prime(self.net_f[p]), np.append(self.h[p], self.h_prime[p + 1])[np.newaxis])
						self.parb_c[p] = self.parb_c[p] * self.y_f[p] + tanh_two_prime(self.net_c[p]) * self.y_n[p]
						self.parb_n[p] = self.parb_n[p] * self.y_f[p] + tanh(self.net_c[p]) * tanh_one_prime(self.net_n[p])
						self.parb_f[p] = self.parb_f[p] * self.y_f[p] + tanh(self.s[p]) * tanh_one_prime(self.net_f[p])
					for q in xrange(len(self.layers) - 2):
						self.w_cc[q] -= eta * self.e_s[q] * self.parw_c[q][:,:self.layers[q]]
						self.w_nc[q] -= eta * self.e_s[q] * self.parw_n[q][:,:self.layers[q]]
						self.w_fc[q] -= eta * self.e_s[q] * self.parw_f[q][:,:self.layers[q]]
						self.w_ch[q] -= eta * self.e_s[q] * self.parw_c[q][:,self.layers[q]:]
						self.w_nh[q] -= eta * self.e_s[q] * self.parw_n[q][:,self.layers[q]:]
						self.w_fh[q] -= eta * self.e_s[q] * self.parw_f[q][:,self.layers[q]:]
						self.b_s[q] -= eta * self.e_s[q] * self.parb_c[q]
						self.b_n[q] -= eta * self.e_s[q] * self.parb_n[q]
						self.b_f[q] -= eta * self.e_s[q] * self.parb_f[q]
					if timer == True:
						stop = time.clock()
						lag = len(datum['x']) * (stop - start)
						minutes = int(lag / 60)
						seconds = int(lag - (minutes * 60))
						timer = False
					print str((k + 1) / len(datum['x']) * 100) + '% - ' + str(minutes) + 'm ' + str(seconds) + 's estimated'
					
	def alpha_loop(self, input, times):
		output = ''
		input = alpha_encode(input)
		for _ in xrange(times):
			self.feed(input)
			output = output + alpha_decode(self.h[-1])
			input = alpha_encode(alpha_decode(self.h[-1]))
		return output