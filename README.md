#Query Based Extractive Summarization Model

To install the necessary tools:

`pip3 install -r requirements.txt`

run this commands to setup the fast text:

`$ cd sent2vec`

`$ sudo make`

`$ pip3 install .`

you have to install the nltk packages. For that run the following commands:

`$ python3`

`>>> import nltk`

`>>> nltk.download()`

Select all and download.

Download CUDA as per your system architecture from this link "https://developer.nvidia.com/cuda-zone" and install it followting the instructions.

download a sent2vec pretrained model from the following link and name it as 'model.bin'
https://github.com/epfml/sent2vec/blob/master/README.md#downloading-sent2vec-pre-trained-models



##Using the class

### Create Object
The main class name is Summarization here. To create an object of this class, just create an object: 

`summary=Summarization(arg_list)`

#### Possible argument lists:

 	1. A string as the source. (Mendatory argument) 

	2. A string for the query. (Mendatory argument)


Optional arguments:
	
	threshold= any decimal value between 0 to 1. This value will be used to compare between two sentences and if they achieve more similarity score than
	the provided threshold, they will go into the same cluster. default value is .2 here.

	alpha = any decimal value between 0 to 1. Default value is .09

	beta = any decimal value between 0 to 1. Default value is .9

	I have used the following formula for sentence ranking: alpha*score1+beta*score2
	where score1=sentence_score_based_on_keywords
	and score2=sentence_similarity_score_with_respect_to_query
	
	also the summation of alpha and beta should be less than 1.

	length= any int value. This will be the maximum length limit of the summary. Default value is 5000 characters.

	length_percent= this is the summary length limit compared to the source document length. Default value is 50%.

	if you provide both the length and length percent, the code will consider the lower one.


### Ranking the sentences based on keyword: 
Call the function by using the following function call:

`summary.rank_sentences_for_keywords()`

### Ranking Sentences based on the similarity with the query: 
For this one, you can either use fasttext or TF_IDF

For fasttext call the following function:

`summary.rank_sentences_for_similarity_fasttext()`

For TF_IDF call the following function:

`summary.rank_sentences_for_similarity_TF_IDF()`

### Clustering the sentences: 

To cluster the sentences call the following function: 

`summary.cluster_sentences()`

### Generating the Summary:

finally to get the summary call the following function: 

`print (summary.generate_summary())`



You can check the example of the function call from the main.py function.




