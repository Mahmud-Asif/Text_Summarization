from summarization import Summarization



if __name__ == '__main__':

    source="Cholesterol is one type of lipid (fat) found in the body. Its \
    an essential part of the membranes surrounding cells and also plays other important roles.\
Cholesterol travels through the bloodstream attached to carrier molecules, including two important \
ones called high-density lipoprotein (HDL) and low-density lipoprotein (LDL). A blood test for cholesterol\
 measures the total amount of lipoprotein-bound cholesterol as well as the relative amounts of HDL and LDL.\
A relatively high level of HDL, also known as good cholesterol, seems to be somewhat protective against \
heart disease. High levels of LDL, also known as bad cholesterol, are associated with an increased risk \
of heart disease.\
Current guidelines issued by the National Cholesterol Education Program establish ranges for cholesterol levels\
 and recommend that regular testing begin at age 20."
    query="alzheimer memory"
    summary=Summarization(source,query)

    summary.rank_sentences_for_keywords()
    summary.rank_sentences_for_similarity_TF_IDF()
    # summary.rank_sentences_for_similarity_fasttext()
    # summary.rank_sentences_for_similarity_universal()
    summary.cluster_sentences()
    
    print (summary.generate_summary())