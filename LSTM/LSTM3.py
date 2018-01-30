from __future__ import division
import numpy as np
import random
import time


def tanh(x):
	return np.tanh(x)
	
def tanh_prime(x):
	return 1 - np.tanh(x) ** 2

def logistic(x):
	return 1 / (1 + np.exp(-x))
	
def logistic_prime(x):
	return logistic(x) * logistic(-x)
	
def cross_entropy_cost(a, y):
	#where a is the output of the network and y is the desired output
	return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))
	
def cross_entropy_cost_prime(a, y):
	return a - y
	
def quadratic_cost(a, y):
	0.5 * np.linalg.norm(a - y) ** 2
	
alpha_cypher = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

def alpha_encode_text(text):
	result = []
	for i in xrange(len(text) - 1):
		datum = {}
		x = [0.0] * len(alpha_cypher)
		x[alpha_cypher.index(text[i])] = 1.0
		y = [0.0] * len(alpha_cypher)
		y[alpha_cypher.index(text[i + 1])] = 1.0
		datum['x'] = x
		datum['y'] = y
		result.append(datum)
	return result

def alpha_encode_data(alpha_data):
	binary_data = []
	for datum in alpha_data:
		encoded_datum = {}
		x = []
		y = []
		for m in datum[0]:
			a = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
			a[alpha_cypher.index(m)] = 1
			x.append(np.array(a)[np.newaxis].T)
		for n in datum[1]:
			a = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
			a[alpha_cypher.index(n)] = 1
			y.append(np.array(a)[np.newaxis].T)
		encoded_datum['x'] = x
		encoded_datum['y'] = y
		binary_data.append(encoded_datum)
	return binary_data

def alpha_encode(alpha):
	binary = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	binary[alpha_cypher.index(alpha)] = 1
	return np.array(binary)[np.newaxis].T
	
def alpha_decode(binary):
	return alpha_cypher[np.where(binary == max(binary))[0][0]]

class LSTM:
	def __init__(self, input, hidden, output):
		#initial weights from hidden to output units and biases of output units
		self.w_oc = np.random.randn(output, hidden)
		self.b_o = np.random.randn(output, 1)
		#initial cell states and biases for gates and cell outputs
		self.s = np.random.randn(hidden, 1)
		self.b_s = np.random.randn(hidden, 1)
		self.b_u = np.random.randn(hidden, 1)
		self.b_f = np.random.randn(hidden, 1)
		self.b_n = np.random.randn(hidden, 1)
		#weights from input to hidden cells and their gates
		self.w_ci = np.random.randn(hidden, input)
		self.w_ui = np.random.randn(hidden, input)
		self.w_fi = np.random.randn(hidden, input)
		self.w_ni = np.random.randn(hidden, input)
		#weights from hidden cells to other hidden cells and their gates
		self.w_ch = np.random.randn(hidden, hidden)
		self.w_uh = np.random.randn(hidden, hidden)
		self.w_fh = np.random.randn(hidden, hidden)
		self.w_nh = np.random.randn(hidden, hidden)
		#outputs for t = -1
		self.h = [np.random.randn(y, 1) for y in [hidden, output]]
		self.h_prime = [np.random.randn(y, 1) for y in [hidden, output]]
		#initial partial derivatives for cell, input gate, and forget gate deltas
		self.parw_c = np.zeros((hidden, input + hidden))
		self.parw_n = np.zeros((hidden, input + hidden))
		self.parw_f = np.zeros((hidden, input + hidden))
		self.parb_c = np.zeros((hidden, 1))
		self.parb_n = np.zeros((hidden, 1))
		self.parb_f = np.zeros((hidden, 1))
		
	def feed(self, input):
		#calculate the input arriving at each cell and gate
		self.input = input
		self.net_c = np.dot(self.w_ci, input) + np.dot(self.w_ch, self.h_prime[0])
		self.net_u = np.dot(self.w_ui, input) + np.dot(self.w_uh, self.h_prime[0])
		self.net_f = np.dot(self.w_fi, input) + np.dot(self.w_fh, self.h_prime[0])
		self.net_n = np.dot(self.w_ni, input) + np.dot(self.w_nh, self.h_prime[0])
		#calculate the output entering the cell state and exiting each gate
		self.y_c = logistic(self.net_c + self.b_s)
		self.y_u = logistic(self.net_u + self.b_u)
		self.y_f = logistic(self.net_f + self.b_f)
		self.y_n = logistic(self.net_n + self.b_n)
		#calculate the state change and output from the cell state
		self.s = self.y_c * self.y_n + self.y_f * self.s
		self.y_s = logistic(self.s + self.b_s)
		#calculate the output leaving the entire cell and save to property h
		self.h_prime[0] = self.h[0]
		self.h[0] = self.y_s * self.y_u
		#calculate the input arriving at the output layer and the output from the output layer
		self.net_o = np.dot(self.w_oc, self.h[0])
		self.y_o = logistic(self.net_o)
		#save the output to property h
		self.h_prime[1] = self.h[1]
		self.h[1] = self.y_o
		
	def train(self, data, epochs, eta):
		first = True
		for i in xrange(epochs):
			start_time = time.clock()
			for j in data:
				for k in xrange(1):
					self.feed(np.array(j['x'])[np.newaxis].T)
					#Apply deltas for the output layer calculated from the errors for each node
					self.err_o = logistic_prime(self.net_o) * (self.y_o - j['y'])
					self.w_oc = self.w_oc - eta * np.dot(self.err_o, self.h[0].T)
					self.b_o = self.b_o - eta * self.err_o
					#Apply deltas for the output gates calculated from the errors for each node
					self.err_u = logistic_prime(self.net_u) * logistic(self.s) * np.dot(self.w_oc.T, self.err_o)
					self.w_ui = self.w_ui - eta * np.dot(self.err_u, self.input.T)
					self.w_uh = self.w_uh - eta * np.dot(self.err_u, self.h_prime[0].T)
					self.b_u = self.b_u - eta * self.err_u
					#Calculate state errors in each cell. Different from other errors!
					self.e_s = self.y_u * logistic_prime(self.s) * np.dot(self.w_oc.T, self.err_o)
					#Calculate new partial derivatives for cell, input gate, and forget gate 
					self.parw_c = self.parw_c * self.y_f + np.dot(logistic_prime(self.net_c) * self.y_n, np.append(self.input, self.h_prime[0])[np.newaxis])
					self.parw_n = self.parw_n * self.y_f + np.dot(logistic(self.net_c) * logistic_prime(self.net_n), np.append(self.input, self.h_prime[0])[np.newaxis])
					self.parw_f = self.parw_f * self.y_f + np.dot(logistic(self.s) * logistic_prime(self.net_f), np.append(self.input, self.h_prime[0])[np.newaxis])
					self.parb_c = self.parb_c * self.y_f + logistic_prime(self.net_c) * self.y_n
					self.parb_n = self.parb_n * self.y_f + logistic(self.net_c) * logistic_prime(self.net_n)
					self.parb_f = self.parb_f * self.y_f + logistic(self.s) * logistic_prime(self.net_f)
					#Apply deltas to cell, input gate, and forget gate calculated form state errors and partial derivatives
					self.w_ci = self.w_ci - eta * self.e_s * self.parw_c[:,:len(self.input)]
					self.w_ni = self.w_ni - eta * self.e_s * self.parw_n[:,:len(self.input)]
					self.w_fi = self.w_fi - eta * self.e_s * self.parw_f[:,:len(self.input)]
					self.w_ch = self.w_ch - eta * self.e_s * self.parw_c[:,len(self.input):]
					self.w_nh = self.w_nh - eta * self.e_s * self.parw_n[:,len(self.input):]
					self.w_fh = self.w_fh - eta * self.e_s * self.parw_f[:,len(self.input):]
					self.b_s = self.b_s - eta * self.e_s * self.parb_c
					self.b_n = self.b_n - eta * self.e_s * self.parb_n
					self.b_f = self.b_f - eta * self.e_s * self.parb_f
			end_time = time.clock()
			if first:
				total_time = epochs * (end_time - start_time) * 1.16
				minutes = int(total_time / 60)
				seconds = int((total_time / 60 - minutes) * 60)
				first = False
			print str((i + 1) / epochs * 100) + '% - estimated time: ' + str(minutes) + 'm ' + str(seconds) + 's - error: ' + str(np.linalg.norm(self.err_o)) 
		print self.h[1]
					
	def alpha_loop(self, input, times):
		self.feed(input)
		output = alpha_decode(self.h[1])
		for l in xrange(times - 1):
			self.feed(alpha_encode(output[-1]))
			output = output + alpha_decode(self.h[1])
		return output