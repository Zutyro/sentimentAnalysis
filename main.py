import nltk
import pandas as pd




def main():
    # nltk.download([
    #             "names",
    #             "stopwords",
    #             "state_union",
    #             "twitter_samples",
    #             "movie_reviews",
    #             "averaged_perceptron_tagger",
    #             "vader_lexicon",
    #             "punkt",])
    # words = [w for w in nltk.corpus.state_union.words() if w.isalpha()]
    # stopwords = nltk.corpus.stopwords.words("english")
    # words = [w for w in words if w.lower() not in stopwords]
    # fd = nltk.FreqDist(words)
    # common = fd.most_common(10)
    # print(common)
    # text = nltk.Text(words)
    # concordance_list = text.concordance_list("america", lines=10)
    # for line in concordance_list:
    #     print(line.line)

    df = pd.read_csv("Sentiment.csv")
    #Odstraňuje nepotřebné sloupce a prázdné řádky
    columns_to_drop = ['id','candidate_confidence','relevant_yn','relevant_yn_confidence','sentiment_confidence','subject_matter_confidence','candidate_gold',
                       'name','relevant_yn_gold','retweet_count','sentiment_gold','subject_matter_gold','tweet_coord','tweet_created','tweet_id',
                       'tweet_location','user_timezone']
    df.drop(columns=columns_to_drop,axis=1,inplace=True)
    df.dropna(inplace=True)

    #Rozděluje dataframe na dataframy podle kandidátů
    dictDf = {}
    for candidate in df.candidate.unique():
        dictDf[candidate] = df.loc[df.candidate == candidate]
    # print(dictDf['Donald Trump'])

    trump_string = ' '.join(dictDf['Donald Trump'].text.values[1:])
    # print(trump_string)
    tk = nltk.TweetTokenizer()
    trump_string_tokens = tk.tokenize(trump_string)
    stopwords = nltk.corpus.stopwords.words('english')
    words = [w for w in trump_string_tokens if w.isalpha() or '#' in w] #Nechává slova s hashtagy kvůli jejich důležitosti
    words = [w for w in words if w.lower() not in stopwords]
    words = [w for w in words if w != 'RT']
    # text = nltk.Text(words)
    # text.concordance("war", lines=10)
    # finder_bi = nltk.collocations.BigramCollocationFinder.from_words(words)
    # most_bi = finder_bi.ngram_fd.most_common(50)
    # count = 0
    # for bi in most_bi:
    #     if bi[1] != count:
    #         count = bi[1]
    #         print(bi)
    finder_tri = nltk.collocations.TrigramCollocationFinder.from_words(words)
    most_tri = finder_tri.ngram_fd.most_common(300)
    count = 0
    for tri in most_tri:
        if tri[1] != count:
            count = tri[1]
            print(tri)
    # finder_quad = nltk.collocations.QuadgramCollocationFinder.from_words(words)
    # print(finder_quad.ngram_fd.most_common(10))







if __name__ == '__main__':
    main()

