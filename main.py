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
    print(dictDf['Donald Trump'])







if __name__ == '__main__':
    main()

