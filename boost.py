"""
Implementation of a spam filter with boosting algorithms. 
"""
import numpy as np

def weight_error(true_labels, pred_labels, weights):
	'''
	Returns weighted error, considers weight vector in calculation. 
	'''	
	n = true_labels.size
	error = 0.0 
	for i in xrange(n):
		if true_labels[i] != pred_labels[i]:
			error += weights[i]
	return error

def weak_clf_plus(data,word_index):
	'''
	Returns vector containing predicted labels of the data.
	Classifier predicts 1 if word i appears in data point x, otherwise 0.
	'''
	predictions = []
	n_sample, n_feat = data.shape
	for i in xrange(n_sample): 
		if data[i][word_index] == 1:
			predictions.append(1)
		else:
			predictions.append(-1)
	return np.array(predictions)

def weak_clf_minus(data,word_index):
	'''
	Returns vector containing predicted labels of the data.
	Classifier predicts 1 if word i does not appear in data point x, otherwise 0. 
	'''
	predictions = []
	n_sample, n_feat = data.shape
	for i in xrange(n_sample): 
		if data[i][word_index] == 0:
			predictions.append(1)
		else:
			predictions.append(-1)
	return np.array(predictions)

def weak_clf_xi_plus(x_i, word_index):
	'''
	Returns single prediction. 
	'''
	if x_i[word_index] == 1:
		return 1
	else: 
		return -1

def weak_clf_xi_minus(x_i, word_index):
	'''
	Returns single prediction
	'''
	if x_i[word_index] == 0:
		return 1
	else: 
		return -1

def final_weak_learner(clf_idx, x_i):
	'''
	Returns single prediction, 1 or -1, for point x_i 
	'''
	if clf_idx > 1530: 
		return weak_clf_xi_minus(x_i, (clf_idx-1531))
	else:
		return weak_clf_xi_plus(x_i,clf_idx)

def boost_algorithm(train_data,labels, vocab, n_rounds):
	'''
	Returns vector containing the weighted errors and vector containing 
	predictions of the weak classifier from t rounds. 
	'''
	
	# Instantiate vector containing weight distribution
	n = labels.size
	vocab_size = vocab.size
	weight_vec = np.array([1.0/n]*n)
	a_vec = []
	clf_indx_vec = []

	print "Number of boosting rounds: %s " % n_rounds
	for t in xrange(n_rounds):
		#print "round %s:" % (t+1)
		error_vec = [] 

		# get weighted error for all weak clfs
		for i in xrange(vocab_size):
			error_vec.append(weight_error(labels, weak_clf_plus(train_data,i),weight_vec))
		for j in xrange(vocab_size):
			error_vec.append(weight_error(labels, weak_clf_minus(train_data,j),weight_vec))
		
		# pick weak learner
		error_vec = np.array(error_vec)
		err = np.amin(error_vec)
		#print "error:%s" %err
		clf_indx = np.argmin(error_vec)
		#print "clfidx: %s" % clf_indx
		a_t = 0.5 * np.log((1.0-err)/err) 
		
		# store for final clf 
		clf_indx_vec.append(clf_indx)
		a_vec.append(a_t)
		
		# reset weight vector
		for z in xrange(n):
			weight_vec[z] = weight_vec[z] * np.exp(-1.0 * a_t * labels[z] * final_weak_learner(clf_indx,train_data[z]))
		weight_vec = weight_vec / np.sum(weight_vec)
		
	return np.array(a_vec), np.array(clf_indx_vec)


def boost_clf(data, a_vec, clf_indx_vec):
	'''
	Return final prediction by calculating weighted majority of classifiers
	in all rounds of boosting. 
	'''
	T = a_vec.size
	final_pred = [] 

	for x in data:
		final_sum = 0
		for t in xrange(T):
			final_sum += a_vec[t] * final_weak_learner(clf_indx_vec[t], x)
		if np.sign(final_sum) == 1:
			final_pred.append(1)
		else: 
			final_pred.append(-1)
	return np.array(final_pred)

def get_words(clf_indexes,word_dict):
	'''
	Returns string of words representing weak classifiers chosen
	'''
	words = []
	n = clf_indexes.size
	for i in xrange(n):
		if clf_indexes[i] > 1530:
			words.append(word_dict[clf_indexes[i]-1531])
		else: 
			words.append(word_dict[clf_indexes[i]])
	return np.array(words,dtype='string')

def calc_error(true_labels, pred_labels):
	'''
	Returns percent error of predicted labels
	'''
	assert true_labels.size == pred_labels.size
	n = true_labels.size

	# Calculate the error
	mis_count = 0.0
	for i in xrange(n):
		if pred_labels[i] != true_labels[i]:
			mis_count += 1.0
	return (mis_count/n)

def main():
	'''
	Runs boosting algorithm on the dataset with designated number of rounds
	'''
	# Parse input
	train = np.genfromtxt('data/train_data.txt')
	test = np.genfromtxt('data/test_data.txt')
	word_dict = np.genfromtxt('dictionary.txt',dtype = 'string')

	# Separate data from labels
	n_feat = word_dict.size
	train_data = train[:,:-1]
	train_labels = train[:,n_feat]
	test_data = test[:,:-1]
	test_labels = test[:,n_feat]

	# Fit classifier, specify number of rounds of boosting
	a_vec, clf_indexes = boost_algorithm(train_data,train_labels,word_dict,10)
	#a_vec, clf_indexes = boost_algorithm(train_data,train_labels,word_dict,2)
	#a_vec, clf_indexes = boost_algorithm(train_data,train_labels,word_dict,3)
	#a_vec, clf_indexes = boost_algorithm(train_data,train_labels,word_dict,7)
	#a_vec, clf_indexes = boost_algorithm(train_data,train_labels,word_dict,15)
	#a_vec, clf_indexes = boost_algorithm(train_data,train_labels,word_dict,20)
	
	# Run classification, print error
	train_err = calc_error(train_labels, boost_clf(train_data, a_vec, clf_indexes))
	test_err = calc_error(test_labels, boost_clf(test_data, a_vec, clf_indexes))

	print "Train Error: ", train_err
	print "Test Error: ", test_err
	print "Buzzwords: "
	print get_words(clf_indexes,word_dict)

if __name__ == '__main__':
	main()