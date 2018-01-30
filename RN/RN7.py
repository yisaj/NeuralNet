from __future__ import division
import numpy as np
import os

top = 0
def if_max(x, y):
	global top
	if max(np.abs(x)) > top:
		top = max(np.abs(x))
		print str(max(np.abs(x))) + ' - ' + y

def f(x):
	return 1.0 / (1.0 + np.exp(-x))
	
def f_prime(x):
	return (1.0 - f(x)) * f(x)

def softmax(x, t):
	return np.exp(x/t) / np.sum(np.exp(x/t))

old_cypher = [' ','1','2','3','4','5','6','7','8','9','0','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','.',',','\'','\"',':',';','-','(',')','/','[00]','[01]','[23]','[03]','[04]','[05]','[06]','[07]','[08]','[09]','[10]','[11]','[12]','[13]','[14]','[15]','[16]','[17]','[18]','[19]','[20]','[21]','[22]','[02]','[24]','[25]','[26]','[27]','[28]']

alpha_cypher = [' ', '.', '<00>', '<01>', '<02>', '<03>', '<04>', '<05>', '<06>', '<07>', '<08>', '<09>', '<10>', '<11>', '<12>', '<13>', '<14>', '<15>', '<16>', '<17>', '<18>', '<19>', '<20>', '<21>', '<22>', '<23>', '<24>', '<25>', '<26>', 'A', 'I', 'M', 'S', 'T', '[00]', '[01]', '[02]', '[03]', '[04]', '[05]', '[06]', '[07]', '[08]', '[09]', '[10]', '[11]', '[12]', '[13]', '[14]', '[15]', '[16]', '[17]', '[18]', '[19]', '[20]', '[21]', '[22]', '[23]', '[24]', '[25]', '[26]', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'y']

def alpha_encode(letter):
	x = [0.0] * 82
	x[alpha_cypher.index(letter)] = 1.0
	return np.array(x)[np.newaxis].T
	
def alpha_decode(letter):
	for i in xrange(len(letter)):
		if letter[i] == 1.0:
			return alpha_cypher[i]

#This is the variable length input output version
def create_alpha_data(array):
	data = []
	for datum in array:
		x = []
		y = []
		for i in xrange(len(datum[0])):
			x.append(alpha_encode(datum[0][i]))
		for j in xrange(len(datum[1])):
			y.append(alpha_encode(datum[1][j]))
		data.append({'x': x, 'y': y})
	return data
	
class RN:
	def __init__(self, inp, hid, out):
		self.inp = inp
		self.hid = hid
		self.out = out
		self.w_ci = np.random.randn(hid, inp) * 0.01
		self.w_cc = np.random.randn(hid, hid) * 0.01
		self.w_oc = np.random.randn(out, hid) * 0.01
		self.b_c = np.zeros((hid, 1))
		self.b_o = np.zeros((out, 1))
		self.wci_hist = np.zeros((self.hid, self.inp))
		self.wcc_hist = np.zeros((self.hid, self.hid))
		self.woc_hist = np.zeros((self.out, self.hid))
		self.bc_hist = np.zeros((self.hid, 1))
		self.bo_hist = np.zeros((self.out, 1))
		self.cprev = np.random.randn(hid, 1)
		self.err = None
			
	def train(self, traindata, epochs, eta):
		loss = 0
		
		for m in xrange(epochs):
			self.cprev = np.zeros((self.hid, 1))

			if m % 100 == 0:
				print "error - " + str(loss)
			loss = 0
			for datum in traindata:
				i, c, o = {}, {}, {}
				c[-1] = self.cprev
				for j in xrange(len(datum['x']) - 1):
					#feed
					i[j] = datum['x'][j]
					c[j] = f(np.dot(self.w_ci, i[j]) + np.dot(self.w_cc, c[j - 1]) + self.b_c)
					o[j] = softmax(np.dot(self.w_oc, c[j]) + self.b_o, 1)
				self.cprev = c[j]
				dw_ci = np.zeros((self.hid, self.inp))
				dw_cc = np.zeros((self.hid, self.hid))
				dw_oc = np.zeros((self.out, self.hid))
				db_c = np.zeros((self.hid, 1))
				db_o = np.zeros((self.out, 1))
				cnext = np.zeros((self.hid, 1))
					#backprop
				for j in reversed(xrange(len(datum['x']) - 1)):
					err = o[j] - datum['y'][j]
					loss += np.sum(np.abs(err))
					dw_oc += np.dot(err, c[j].T)
					db_o += err
					err = (np.dot(self.w_oc.T, err) + cnext) * (1 - c[j]) * c[j]
					db_c += err
					dw_ci += np.dot(err, i[j].T)
					dw_cc += np.dot(err, c[j - 1].T)
					cnext = np.dot(self.w_cc.T, err)
				for param in [dw_ci, dw_cc, dw_oc, db_c, db_o]:
					np.clip(param, -5, 5, out=param)
				
				self.wci_hist += dw_ci ** 2
				self.wcc_hist += dw_cc ** 2
				self.woc_hist += dw_oc ** 2
				self.bc_hist += db_c ** 2
				self.bo_hist += db_o ** 2
				self.w_ci -= dw_ci * eta / np.sqrt(self.wci_hist + 1e-8)
				self.w_cc -= dw_cc * eta / np.sqrt(self.wcc_hist + 1e-8)
				self.w_oc -= dw_oc * eta / np.sqrt(self.woc_hist + 1e-8)
				self.b_c -= db_c * eta / np.sqrt(self.bc_hist + 1e-8)
				self.b_o -= db_o * eta / np.sqrt(self.bo_hist + 1e-8)
				
	def alpha_loop(self, input, times, temperature):
		result = ""
		i, c, o, l = {}, {}, {}, {}
		c[-1] = self.cprev
		for j in xrange(len(input)):
			c[j] = f(np.dot(self.w_ci, input[j]) + np.dot(self.w_cc, c[j - 1]) + self.b_c)
			o[j] = softmax(np.dot(self.w_oc, c[j]) + self.b_o, 1)
			l[j] = o[j]
			i[j + 1] = o[j]
		for j in xrange(len(input), times + len(input)):
			result += alpha_cypher[np.random.choice(range(self.inp), p = l[len(l) - 1].ravel())]
			c[j] = f(np.dot(self.w_ci, i[j]) + np.dot(self.w_cc, c[j - 1]) + self.b_c)
			o[j] = softmax(np.dot(self.w_oc, c[j]) + self.b_o, 1)
			l[j] = o[j]
			i[j + 1] = o[j]
		return result

def process_article(directory):
	articles = []
	for subdir, dirs, files in os.walk(directory):
		n = 0
		for file in files:
			article = open(os.path.join(directory, file), 'r').read()
			article = article.replace(',', '')
			article = article.replace('"', '')
			article = article.replace('\'', '')
			article = article.replace('?', '')
			article = article.replace('!', '')
			article = article.replace(';', '')
			article = article.replace(':', '')
			article = article.split("\n\n")
			article[0] = article[0][6:]
			article[-1] = article[-1][8:]
			for i in xrange(len(article)):
				article[i] = article[i].split('. ')
				for j in xrange(len(article[i])):
					if '.' in article[i][j]:
						article[i][j] = article[i][j][:-1]
					article[i][j] = article[i][j].split(' ')
			articles.append(article)
			n += 1
	return articles

filler_words = ["for", "and", "nor", "but", "or", "yet", "so", "after", "although", "as", "as", "as long as", "because", "before", "even", "though", "if", "once", "provided", "so", "since", "that", "though", "till", "unless", "until", "when", "what", "whenever", "wherever", "while", "whether", "accordingly", "also", "anyway", "besides", "consequently", "finally", "for example", "for instance", "further", "again", "anyway", "meanwhile", "accordingly", "also", "however", "next", "consequently", "besides", "instead", "then", "hence", "finally", "nevertheless", "now", "henceforth", "further", "otherwise", "thereafter", "therefore", "furthermore", "contrarily", "thus", "moreover", "conversely", "incidentally", "nonetheless", "subsequently", "namely", "likewise", "indeed", "still", "specifically", "similarly", "nevertheless", "undoubtedly", "certainly", "above", "across", "after", "at", "around", "before", "behind", "below", "beside", "between", "by", "down", "during", "from", "for", "in", "inside", "onto", "of", "off", "on", "out", "through", "to", "under", "up", "with", "who", "that", "which", "whom", "whose", "the", "a", "an", "be", "can", "could", "dare", "do", "have", "may", "might", "must", "need", "ought", "shall", "should", "will", "would", "this", "that", "these", "those", "my", "your", "yours", "his", "her", "hers", "its", "our", "their", "theirs", "whose", "much", "many", "few", "more", "most", "least", "little", "less", "fewer", "a lot", "plenty", "great deal", "several", "couple", "each", "every", "neither", "all", "both", "I", "you", "he", "she", "they", "them", "we", "us", "been", "being", "am", "are", "is", "was", "were", "had", "have", "has", "it", "there", "due"]

def cut_words(articles):
	for a in xrange(len(articles)):
		for p in xrange(len(articles[a])):
			for s in xrange(len(articles[a][p])):
				for word in filler_words:
					for i in xrange(articles[a][p][s].count(word)):
						if word in articles[a][p][s]:
							articles[a][p][s].remove(word)
					for i in xrange(articles[a][p][s].count(word.title())):
						if word.title() in articles[a][p][s]:
							articles[a][p][s].remove(word.title())
	return articles
	
def create_vocabs(articles):
	vocabs = []
	for article in articles:
		vocab = []
		article = np.sum(np.sum(article))
		for word in article:
			if not word in vocab:
				vocab.append(word)
		vocabs.append(vocab)
	return vocabs
	
def create_summary_text(articles, vocabs):
	#articles should be the full text and vocabs should be the vocabulary of articles with article unspecific words cut out
	result = [[],[],[]]
	for i in xrange(len(articles)):
		summarywords = []
		textarray = []
		wordpos = []
		wordstext = []
		summary = articles[i][-1][0]
		for word in vocabs[i]:
			if word in summary:
				summarywords.append(word)
		c = 0
		for j in xrange(len(summary)):
			textarray.append(" ")
			if summary[j] in summarywords:
				if c < 10:
					wordpos.append("<0" + str(c) + ">")
					wordstext.append(summary[j])
				else:
					wordpos.append("<" + str(c) + ">")
					wordstext.append(summary[j])
				if summarywords.index(summary[j]) < 10:
					textarray.append("[0" + str(summarywords.index(summary[j])) + "]")
				else:
					textarray.append("[" + str(summarywords.index(summary[j])) + "]")
				c += 1
			else:
				for letter in summary[j]:
					textarray.append(letter)
			
		result[1].append(wordpos)
		textarray.append('.')
		wordpos.append('.')
		result[0].append(textarray)
		result[2].append(wordstext)
	return result

def create_summary_data(summarytext):
	data = []
	for i in xrange(int(np.ceil(len(summarytext) / 25))):
		x = []
		y = []
		for j in xrange(i * 25, (i + 1) * 25 - 1):
			if j + 1 >= len(summarytext):
				x.append(alpha_encode(summarytext[j - len(summarytext)]))
				y.append(alpha_encode(summarytext[j + 1 - len(summarytext)]))
			else:
				x.append(alpha_encode(summarytext[j]))
				y.append(alpha_encode(summarytext[j + 1]))
		data.append({'x':x, 'y':y})
	return data