import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def cosine_similarities(tfidf_matrix, sentence1, sentence2):
	return cosine_similarity(tfidf_matrix[sentence1:sentence1+1], tfidf_matrix[sentence2:sentence2+1])


class Cluster():

	def merge(self, ranked_sentence):
		merged_sentences=[]
		for i in range (0, len(ranked_sentence)):
			merged_sentences.append(str(ranked_sentence[i][2]))
		return merged_sentences

	def TF_ID_Vectorization(self, documents, threshold):

		tfidf_matrix=[]
		cluster_mark=[]
		cluster_rank=[]
		cluster_list=[]		
		check_tfidf_matrix=[]
		cluster_mark=np.full((len(documents),5),-1.0)
		check_tfidf_matrix=np.zeros(len(documents))
		number_of_cluster=len(documents)
		sentence_per_cluster=len(documents)
		tfidf_vectorizer = TfidfVectorizer()

		#Creating TF-IDF Matrix
		tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

		for i in range (0,tfidf_matrix.shape[0]):
			if check_tfidf_matrix[i]!=0:
				continue
			max_val=0.0
			best_match=0;
			for j in range (0, i):
				if check_tfidf_matrix[j]!=0:
					continue

				x=cosine_similarities(tfidf_matrix,i,j)
				if x>.95:
					continue
				if x>max_val:
					max_val=x
					best_match=j
			
			cluster_mark[i][1]=max_val
			cluster_mark[i][2]=best_match

			###Add sentence to the existing Cluster
			if max_val>=threshold:
				cluster_mark[i][0]=int(cluster_mark[best_match][0])
				cluster_list[int(cluster_mark[i][0])].append(documents[i])

			###Create a new cluster with the sentence
			else:
				cluster_mark[i][0]=len(cluster_list)
				cluster_list.append([])
				cluster_list[len(cluster_list)-1].append(documents[i])

		return cluster_list

