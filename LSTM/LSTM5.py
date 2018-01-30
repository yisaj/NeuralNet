from __future__ import division
import numpy as np
import random
import time

top = 0
def if_max(x, y):
	global top
	if max(np.abs(x)) >= top:
		top = max(np.abs(x))
		print str(max(np.abs(x))) + ' - ' + y

def f(x):
	if_max(x, 'f')
	return np.clip(1 / (1 + np.exp(-x)), 0.0000000001, 0.9999999999)
	
def f_prime(x):
	if_max(x, 'fp')
	return np.clip((1 - (1 / (1 + np.exp(-x)))) * (1 / (1 + np.exp(-x))), 0.25, 0.9999999999)
	
def g(x):
	if_max(x, 'g')
	return np.clip(4 / (1 + np.exp(-x)) - 2, -1.9999999999, 1.9999999999)
	
def g_prime(x):
	if_max(x, 'gp')
	return np.clip((1 - (1 / (1 + np.exp(-x)))) * (1 / (1 + np.exp(-x))) * 4, 1, 0.0000000001)

def h(x):
	if_max(x, 'h')
	return np.clip(2 / (1 + np.exp(-x)) - 1, -0.9999999999, 0.9999999999)
	
def h_prime(x):
	if_max(x, 'hp')
	return np.clip((1 - (1 / (1 + np.exp(-x)))) * (1 / (1 + np.exp(-x))) * 2, 0.5, 0.0000000001)

alpha_cypher = [' ','1','2','3','4','5','6','7','8','9','0','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','.',',','\'','\"',':',';','-','(',')','!','?']

binary_cypher = [
[0,0,0,0,0,0,0],[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1],[1,1,0,0,0,0,0],[1,0,1,0,0,0,0],[1,0,0,1,0,0,0],
[1,0,0,0,1,0,0],[1,0,0,0,0,1,0],[1,1,1,0,0,1,1],[1,1,1,0,0,0,0],[1,1,0,1,0,0,0],[1,1,0,0,1,0,0],[1,1,0,0,0,1,0],[1,1,0,0,0,0,1],[1,1,1,1,0,0,0],[1,1,1,0,1,0,0],[1,1,1,0,0,1,0],
[0,1,0,0,1,1,0],[1,1,1,1,1,0,0],[1,1,1,1,0,1,0],[1,1,1,1,0,0,1],[1,1,1,1,1,1,0],[1,1,1,1,1,0,1],[1,1,1,1,1,1,1],[0,1,1,1,1,1,1],[1,0,1,1,1,1,1],[1,1,0,1,1,1,1],[1,1,1,0,1,1,1],
[1,1,1,1,0,1,1],[0,0,1,1,1,1,1],[0,1,0,1,1,1,1],[0,1,1,0,1,1,1],[0,1,1,1,0,1,1],[0,1,1,1,1,0,1],[0,1,1,1,1,1,0],[0,0,0,1,1,1,1],[0,0,1,0,1,1,1],[0,0,1,1,0,1,1],[0,0,1,1,1,0,1],
[0,0,1,1,1,1,0],[0,0,0,0,1,1,1],[0,0,0,1,0,1,1],[0,0,0,1,1,0,1],[0,0,0,1,1,1,0],[0,0,0,0,0,1,1],[0,0,0,0,1,0,1],[0,0,0,0,1,1,0],[0,0,1,1,0,0,0],[0,0,0,1,1,0,0],[0,1,0,0,0,1,0],
[0,1,0,1,0,0,0],[0,0,0,1,0,1,0],[1,1,0,0,1,1,0],[0,1,1,0,1,0,1],[0,1,0,1,1,0,0],[1,0,1,1,1,0,0],[0,0,1,1,0,1,0],[1,0,0,0,0,0,1],[1,1,0,0,1,1,1],[1,0,0,0,0,1,1],[1,1,1,0,0,0,1],
[1,0,0,0,1,1,1],[1,0,0,1,0,1,1],[1,1,0,1,0,0,1],[0,1,0,1,0,1,1],[1,1,0,1,0,1,0],[0,1,1,1,1,0,0],[0,1,0,1,1,1,0],[0,1,1,1,0,1,0]
]

def alpha_encode_bible_compact(string):
	x = []
	y = []
	for i in xrange(len(string) - 1):
		x.append(np.array(binary_cypher[alpha_cypher.index(string[i])])[np.newaxis].T)
		y.append(np.array(binary_cypher[alpha_cypher.index(string[i + 1])])[np.newaxis].T)
	return [{'x': x, 'y': y}]
	
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
			a = [0] * 74
			a[alpha_cypher.index(x)] = 1
			set['x'].append(np.array(a)[np.newaxis].T)
		for y in datum[1]:
			b = [0] * 74
			b[alpha_cypher.index(y)] = 1
			set['y'].append(np.array(b)[np.newaxis].T)
		result.append(set)
	return result
	
def alpha_decode(binary):
	return alpha_cypher[binary.tolist().index(max(binary).tolist())]

def alpha_encode(alpha):
	result = [0] * 74
	result[alpha_cypher.index(alpha)] = 1
	return np.array(result)[np.newaxis].T
	
class LSTM:
	def __init__(self, layers):
		self.layers = layers
		#weights from last layer of hidden cells to output layer
		self.w_oc = np.random.randn(layers[-1], layers[-2])
		self.b_o = np.random.randn(layers[-1], 1)
		#initial cell states and biases for each gate and cell output
		self.s = [np.random.randn(y, 1) for y in layers[1:-1]]
		self.s_prime = [np.random.randn(y, 1) for y in layers[1:-1]]
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
		#intermediate values
		self.net_c = [np.zeros((y, 1)) for y in layers[1:-1]]
		self.net_u = [np.zeros((y, 1)) for y in layers[1:-1]]
		self.net_f = [np.zeros((y, 1)) for y in layers[1:-1]]
		self.net_n = [np.zeros((y, 1)) for y in layers[1:-1]]
		self.y_c = [np.zeros((y, 1)) for y in layers[1:-1]]
		self.y_u = [np.zeros((y, 1)) for y in layers[1:-1]]
		self.y_f = [np.zeros((y, 1)) for y in layers[1:-1]]
		self.y_n = [np.zeros((y, 1)) for y in layers[1:-1]]
		self.y_s = [np.zeros((y, 1)) for y in layers[1:-1]]
		
	def feed(self, input):
		self.h_prime[0] = self.h[0]
		self.h[0] = input
		for j in xrange(len(self.layers) - 2):
			self.h_prime[j + 1] = self.h[j + 1]
			self.net_c[j] = np.dot(self.w_cc[j], self.h[j]) + np.dot(self.w_ch[j], self.h_prime[j + 1])
			self.net_u[j] = np.dot(self.w_uc[j], self.h[j]) + np.dot(self.w_uh[j], self.h_prime[j + 1])
			self.net_f[j] = np.dot(self.w_fc[j], self.h[j]) + np.dot(self.w_fh[j], self.h_prime[j + 1])
			self.net_n[j] = np.dot(self.w_nc[j], self.h[j]) + np.dot(self.w_nh[j], self.h_prime[j + 1])
			
			self.y_c[j] = g(self.net_c[j] + self.b_s[j])
			self.y_u[j] = f(self.net_u[j] + self.b_u[j])
			self.y_f[j] = f(self.net_f[j] + self.b_f[j])
			self.y_n[j] = f(self.net_n[j] + self.b_n[j])
			
			self.s_prime[j] = self.s[j]
			self.s[j] = self.y_c[j] * self.y_n[j] + self.s[j] * self.y_f[j]
			self.y_s[j] = h(self.s[j])
			
			self.h[j + 1] = self.y_s[j] * self.y_u[j]
		self.net_o = np.dot(self.w_oc, self.h[-2])
		self.y_o = f(self.net_o + self.b_o)
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
					
					self.err_o = self.y_o - datum['y'][k]
					self.w_oc -= eta * np.dot(self.err_o, self.h[-2].T)
					self.b_o -= eta * self.err_o
					
					self.err_u = [f_prime(self.net_u[-1]) * h(self.s[-1]) * np.dot(self.w_oc.T, self.err_o)]
					for l in xrange(len(self.layers) - 3):
						self.err_u.insert(0, np.dot(self.w_uc[-l - 1].T, self.err_u[-l - 1]) * f_prime(self.net_u[-l - 2]) * h(self.s[-l - 2]))
					for m in xrange(len(self.layers) - 2):
						self.w_uc[m] -= eta * np.dot(self.err_u[m], self.h[m].T)
						self.w_uh[m] -= eta * np.dot(self.err_u[m], self.h_prime[m + 1].T)
						self.b_u[m] -= eta * self.err_u[m]
						
					self.e_s = [self.y_u[-1] * h_prime(self.s[-1]) * np.dot(self.w_oc.T, self.err_o)]
					for n in xrange(len(self.layers) - 3):
					
						self.e_s.insert(0, (np.dot(self.w_cc[-n - 1].T, self.e_s[-n - 1] * self.y_n[-n - 1] * g_prime(self.net_c[-n - 1])) + np.dot(self.w_nc[-n - 1].T, self.e_s[-n - 1] * g(self.net_c[-n - 1]) * f_prime(self.net_n[-n - 1])) + np.dot(self.w_fc[-n - 1].T, self.e_s[-n - 1] * f_prime(self.net_f[-n - 1]) * self.s_prime[-n - 1])) * h_prime(self.s[-n - 2]) * self.y_u[-n - 2])
						#self.e_s.insert(0, self.y_u[-n - 2] * h_prime(self.s[-n - 2] * np.dot(self.w_uc[-n - 1].T, self.err_u[-n - 1])))
					for p in xrange(len(self.layers) - 2):
						self.parw_c[p] = self.parw_c[p] * self.y_f[p] + np.dot(g_prime(self.net_c[p]) * self.y_n[p], np.append(self.h[p], self.h_prime[p + 1])[np.newaxis])
						self.parw_n[p] = self.parw_n[p] * self.y_f[p] + np.dot(g(self.net_c[p]) * f_prime(self.net_n[p]), np.append(self.h[p], self.h_prime[p + 1])[np.newaxis])
						self.parw_f[p] = self.parw_f[p] * self.y_f[p] + np.dot(h(self.s[p]) * f_prime(self.net_f[p]), np.append(self.h[p], self.h_prime[p + 1])[np.newaxis])
						self.parb_c[p] = self.parb_c[p] * self.y_f[p] + g_prime(self.net_c[p]) * self.y_n[p]
						self.parb_n[p] = self.parb_n[p] * self.y_f[p] + g(self.net_c[p]) * f_prime(self.net_n[p])
						self.parb_f[p] = self.parb_f[p] * self.y_f[p] + h(self.s[p]) * f_prime(self.net_f[p])
					for q in xrange(len(self.layers) - 2):
						self.w_cc[q] += eta * self.e_s[q] * self.parw_c[q][:,:self.layers[q]]
						self.w_nc[q] += eta * self.e_s[q] * self.parw_n[q][:,:self.layers[q]]
						self.w_fc[q] += eta * self.e_s[q] * self.parw_f[q][:,:self.layers[q]]
						self.w_ch[q] += eta * self.e_s[q] * self.parw_c[q][:,self.layers[q]:]
						self.w_nh[q] += eta * self.e_s[q] * self.parw_n[q][:,self.layers[q]:]
						self.w_fh[q] += eta * self.e_s[q] * self.parw_f[q][:,self.layers[q]:]
						self.b_s[q] += eta * self.e_s[q] * self.parb_c[q]
						self.b_n[q] += eta * self.e_s[q] * self.parb_n[q]
						self.b_f[q] += eta * self.e_s[q] * self.parb_f[q]
					if timer == True:
						stop = time.clock()
						lag = len(datum['x']) * (stop - start)
						minutes = int(lag / 60)
						seconds = int(lag - (minutes * 60))
						timer = False
					print str((k + 1) / len(datum['x']) * 100) + '% - ' + str(minutes) + 'm ' + str(seconds) + 's estimated'
	
	def alpha_loop_compact(self, input, times):
		result = ''
		input = np.array(binary_cypher[alpha_cypher.index(input)])[np.newaxis].T
		for i in xrange(times):
			self.feed(input)
			if np.round(self.h[-1]).T.tolist()[0] in binary_cypher:
				result += alpha_cypher[binary_cypher.index(np.round(self.h[-1]).T.tolist()[0])]
				input = np.round(self.h[-1])
			else:
				result += '_'
				input = self.h[-1]
		return result
				
	def alpha_loop(self, input, times):
		result = ''
		input = alpha_encode(input)
		for i in xrange(times):
			self.feed(input)
			result += alpha_decode(self.h[-1])
			input = alpha_encode(alpha_decode(self.h[-1]))
		return result