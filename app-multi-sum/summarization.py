import sent2vec
from keyword_score_generate import Score
from similarity_score import Similarity
from cluster import Cluster
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

score=Score()
cluster=Cluster()
sim=Similarity()

def cosine_similarities(tfidf_matrix, sentence1, sentence2):
    return cosine_similarity(tfidf_matrix[sentence1:sentence1+1], tfidf_matrix[sentence2:sentence2+1])

class Summarization():

    def __init__(self, source, query, threshold=.4, alpha=.09, beta=.9, summary_length_limit=5000, summary_length_percentage=50):
        self.source_text = source
        self.query_text = query
        self.similarity_threshold = threshold
        self.alpha = alpha
        self.beta = beta
        self.summary_length_limit = summary_length_limit
        self.summary_length_percentage = summary_length_percentage
        self.length_source=len(self.source_text)
        if (alpha+beta)>1:
            print ("Wrong alpha beta value combination. alpha+beta should be equal to 1")
            sys.exit()
        self.summary_length_limit=min(self.summary_length_limit,(self.length_source*self.summary_length_percentage)/100)
  

    def rank_sentences_for_keywords(self):
        self.ranked_sentences,self.original_sentence_rank=score.rank_sentence(self.source_text)
        print ("\n\nSentences ranked based on keywords\n\n")

    def rank_sentences_for_similarity_fasttext(self):
        model = sent2vec.Sent2vecModel()
        model.load_model('model.bin')
        self.ranked_sentences =sim.ranked_by_similarity_fasttext(self.ranked_sentences, self.query_text, self.alpha, self.beta,model)
        self.ranked_sentences=sorted(self.ranked_sentences, key=lambda x:(x[1]),reverse=True)
        print ("Sentences ranked based on the Similarity with the query\n")  

    def rank_sentences_for_similarity_universal(self):
        self.ranked_sentences =sim.ranked_by_similarity_universal(self.ranked_sentences, self.query_text, self.alpha, self.beta)
        self.ranked_sentences=sorted(self.ranked_sentences, key=lambda x:(x[1]),reverse=True)
        print ("Sentences ranked based on the Similarity with the query\n")  

    def rank_sentences_for_similarity_TF_IDF(self):
        # self.ranked_sentences.append([0,0,self.query_text])
        merged_sentences=cluster.merge(self.ranked_sentences)
        self.ranked_sentences =sim.ranked_by_similarity_TF_IDF(merged_sentences)
        self.ranked_sentences=sorted(self.ranked_sentences, key=lambda x:(x[1]),reverse=True)
        print ("Sentences ranked based on the Similarity with the query\n")  

    def cluster_sentences(self):
        self.merged_sentences=cluster.merge(self.ranked_sentences)
        self.clustered_sentences=cluster.TF_ID_Vectorization(self.merged_sentences,float(self.similarity_threshold))
        print ("Clustered the sentences\n")

    def generate_summary(self):
        summary=''
        summary_temp=''
        temp_sum=[]
        for i in range (0,len(self.clustered_sentences)):
            if len(summary)+len(self.clustered_sentences[i][0])<=self.summary_length_limit:
                summary_temp=summary_temp+self.clustered_sentences[i][0]
                temp_sum.append([self.clustered_sentences[i][0]])

        for i in range(0, len(temp_sum)):
            for j in range (0, len(self.original_sentence_rank)):
                if str(temp_sum[i][0])==str(self.original_sentence_rank[j][1]):
                    temp_sum[i].append(self.original_sentence_rank[j][0])
                    break

        temp_sum=sorted(temp_sum,key=lambda x:(x[1]))
        for i in range(0, len(temp_sum)):
            summary=summary+str(temp_sum[i][0])

        print ("Summary Generated for query: "+self.query_text)
        return summary


 
