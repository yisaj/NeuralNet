def separate(article):
	article = article.lower()
	article = article.replace("\"", "")
	article = article.replace(",", "")
	article = article.replace("-", "")
	article = article.split("\n\n")
	for i in xrange(len(article)):
		article[i] = article[i].split(". ")
		for j in xrange(len(article[i])):
			article[i][j] = article[i][j].replace("\n", "")
			article[i][j] = article[i][j].replace(".", "")
			article[i][j] = article[i][j].split(" ")
	return article
	
def process(article):
	vocab = {}
	for i in xrange(len(article)):
		for j in xrange(len(article[i])):
			for k in xrange(len(article[i][j]])):
				if (not article[i][j][k] in vocab):
					vocab[article[i][j][k]] = {'length': len(article[i][j][k]), 'd_index': i, 'p_index': j, 's_index': k, 'count': 1}