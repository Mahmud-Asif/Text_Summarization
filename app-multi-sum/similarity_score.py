import tensorflow as tf 
import tensorflow_hub as hub 
import numpy as np
import os
import sent2vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from numba import autojit, prange
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def cosine_similarities(sentence1, sentence2):
	return cosine_similarity(sentence1,sentence2)

def universal_embedding(sentence, query):
	module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
	embed = hub.Module(module_url)
	messages =[query]
	messages1 = [sentence]
	
	with tf.Session() as session:
		session.run([tf.global_variables_initializer(), tf.tables_initializer()])
		message_embeddings = session.run(embed(messages))
		message_embeddings1 = session.run(embed(messages1))
		sim_matrix = np.inner(message_embeddings, message_embeddings1)
		session.close()
	  	
	tf.reset_default_graph()
	return sim_matrix

def fasttext_embedding(sentence, query,model):
	messages =query
	messages1 = sentence
	
	with tf.Session() as session:
		session.run([tf.global_variables_initializer(), tf.tables_initializer()])
		message_embeddings = model.embed_sentence(messages)
		message_embeddings1 = model.embed_sentence(messages1)
		sim_matrix = np.inner(message_embeddings, message_embeddings1)
		session.close()
	  	
	tf.reset_default_graph()
	return sim_matrix
	  

class Similarity():
	@autojit
	def ranked_by_similarity_fasttext(self, doc, query, alpha, beta,model):
		for i in range (0,len(doc)):
			sentence=doc[i][2]
			sim_score=fasttext_embedding(str(sentence), str(query),model)
			doc[i][1]=alpha*doc[i][1]+beta*sim_score[0][0]
		return doc

	@autojit
	def ranked_by_similarity_universal(self, doc, query, alpha, beta):
		for i in range (0,len(doc)):
			sentence=doc[i][2]
			sim_score=universal_embedding(str(sentence), str(query))
			doc[i][1]=alpha*doc[i][1]+beta*sim_score[0][0]

		return doc

	@autojit
	def ranked_by_similarity_TF_IDF(self, doc):		
		tfidf_vectorizer = TfidfVectorizer()
		tfidf_matrix = tfidf_vectorizer.fit_transform(doc)
		doc_sort=[]
		for i in range (0,tfidf_matrix.shape[0]-1):
			sim_score=cosine_similarities(tfidf_matrix[i:i+1], tfidf_matrix[tfidf_matrix.shape[0]-1:tfidf_matrix.shape[0]])
			doc_sort.append([i,sim_score,str(doc[i])])

		return doc_sort

    

