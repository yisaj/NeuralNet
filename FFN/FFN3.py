raw_data = [
['abcd','bcde'],
['efgh','fghi'],
['ijkl','jklm'],
['mnop','nopq'],
['qrst','rstu'],
['uvwx','vwxy'],
['yzab','zabc'],
]

import numpy as np
import os
import pickle
os.chdir('C:\Users\Azerk\Desktop\Dropbox\Workshop\Python')
import LSTM7
biblenet = pickle.load(open('biblenet', 'r'))
bible = LSTM7.alpha_encode_bible(open('11.txt', 'r').read().decode('utf-8-sig').encode('utf-8'))

for i in xrange(500):
	if np.abs((net.net_c[0] + net.b_s[0])[i]) > 1000:
		print net.net_c[0][i] + net.b_s[0][i]
		
for i in xrange(len(net.w_ch[0])):
	if max(np.abs(net.w_ch[0][i])) > 10:
		print max(np.abs(net.w_ch[0][i]))
		
open('12.txt', 'r').read().decode('utf-8-sig').encode('utf-8') + open('13.txt', 'r').read().decode('utf-8-sig').encode('utf-8') + open('14.txt', 'r').read().decode('utf-8-sig').encode('utf-8') + open('15.txt', 'r').read().decode('utf-8-sig').encode('utf-8') + open('16.txt', 'r').read().decode('utf-8-sig').encode('utf-8') + open('17.txt', 'r').read().decode('utf-8-sig').encode('utf-8') + open('18.txt', 'r').read().decode('utf-8-sig').encode('utf-8') + open('19.txt', 'r').read().decode('utf-8-sig').encode('utf-8') + open('20.txt', 'r').read().decode('utf-8-sig').encode('utf-8') + open('21.txt', 'r').read().decode('utf-8-sig').encode('utf-8') + open('22.txt', 'r').read().decode('utf-8-sig').encode('utf-8') + open('23.txt', 'r').read().decode('utf-8-sig').encode('utf-8') + open('24.txt', 'r').read().decode('utf-8-sig').encode('utf-8') + open('25.txt', 'r').read().decode('utf-8-sig').encode('utf-8') + open('26.txt', 'r').read().decode('utf-8-sig').encode('utf-8') + open('27.txt', 'r').read().decode('utf-8-sig').encode('utf-8')

for i in xrange(53):
	bible += open(str(12 + i) + '.txt', 'r').read().decode('utf-8-sig').encode('utf-8')