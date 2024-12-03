import nltk




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
    words = [w for w in nltk.corpus.state_union.words() if w.isalpha()]
    stopwords = nltk.corpus.stopwords.words("english")
    words = [w for w in words if w.lower() not in stopwords]
    fd = nltk.FreqDist(words)
    common = fd.most_common(10)
    print(common)
    text = nltk.Text(words)
    concordance_list = text.concordance_list("america", lines=10)
    for line in concordance_list:
        print(line.line)




if __name__ == '__main__':
    main()

