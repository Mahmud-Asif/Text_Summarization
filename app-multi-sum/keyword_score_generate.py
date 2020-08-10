from nltk.tokenize import sent_tokenize
import string
from os import path
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import rake
import nltk.data

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
rake_object = rake.Rake()

class Score():

	def gen_keywords(self, text):
		text=str(text)
		text=text.lower()
		keywords=rake_object.get_phrases(text)
		return keywords

	def rank_sentence(self, text):
		text=str(text)		
		text1 = sent_tokenize(str(text))		
		keywords=self.gen_keywords(text)
		text_main=text1
		sentence_score=[0.0]*(len(text1)+100)

		for i in range (0,len(keywords)):
			score=keywords[i][1]
			phrase=keywords[i][0]			
			line_no=0			
			j=0
			for line in text1:
				line=str(line)
				line=line.lower()
				if line.count(phrase)>0:
					pp=line.count(phrase)
					sentence_score[j]=sentence_score[j]+(score*pp)
				j+=1
				line_no+=1

			text=text.replace(phrase,' ')

		ranked_sentence=[]
		original_sentence_rank=[]
		for i in range(0,len(text1)):
			if sentence_score[i]>0:
				ranked_sentence.append([i,sentence_score[i],text_main[i]])
				original_sentence_rank.append([i,text_main[i]])


		ranked_sentence=sorted(ranked_sentence, key=lambda x:(x[1]),reverse=True)
		return ranked_sentence,original_sentence_rank
