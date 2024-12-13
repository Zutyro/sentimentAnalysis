from traceback import print_tb

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd




def main():
    # nltk.download([
    #             "stopwords",
    #             "averaged_perceptron_tagger",
    #             "vader_lexicon",
    #             "punkt",])

    df = pd.read_csv("Sentiment.csv")
    #Odstraňuje nepotřebné sloupce a prázdné řádky
    columns_to_drop = ['id','candidate_confidence','relevant_yn','relevant_yn_confidence','sentiment_confidence','subject_matter_confidence','candidate_gold',
                       'name','relevant_yn_gold','retweet_count','sentiment_gold','subject_matter_gold','tweet_coord','tweet_created','tweet_id',
                       'tweet_location','user_timezone']
    df.drop(columns=columns_to_drop,axis=1,inplace=True)
    df.dropna(inplace=True)

    #Rozděluje dataframe na dataframy podle kandidátů
    dfDict = {}
    for candidate in df.candidate.unique():
        dfDict[candidate] = df.loc[df.candidate == candidate]

    subjectsDict = {}
    for dfKey, dfValue in dfDict.items():
        df_string = ' '.join(dfValue.text.values[1:])
        tk = nltk.TweetTokenizer()
        string_tokens = tk.tokenize(df_string)
        stopwords = nltk.corpus.stopwords.words('english')
        words = [w for w in string_tokens if
                 w.isalpha() or '#' in w]  # Nechává slova s hashtagy kvůli jejich důležitosti
        words = [w for w in words if w.lower() not in stopwords]
        words = [w for w in words if w != 'RT']
        finder_bi = nltk.collocations.BigramCollocationFinder.from_words(words)
        most_bi = finder_bi.ngram_fd.most_common(100)
        count = 0
        bi_subjects = []
        for bi in most_bi:
            if bi[1] != count: #Snaží se zabránit více kolokacím ze stejné věty způsobené retweetem
                count = bi[1]
                bi_subjects.append(bi)
        finder_tri = nltk.collocations.TrigramCollocationFinder.from_words(words)
        most_tri = finder_tri.ngram_fd.most_common(100)
        count = 0
        tri_subjects = []
        for tri in most_tri:
            if tri[1] != count: #Snaží se zabránit více kolokacím ze stejné věty způsobené retweetem
                count = tri[1]
                tri_subjects.append(tri)
        finder_quad = nltk.collocations.QuadgramCollocationFinder.from_words(words)
        most_quad = finder_quad.ngram_fd.most_common(100)
        count = 0
        quad_subjects = []
        for quad in most_quad:
            if quad[1] != count:  #Snaží se zabránit více kolokacím ze stejné věty způsobené retweetem
                count = quad[1]
                quad_subjects.append(quad)
        subjectsDict[dfKey] = [bi_subjects, tri_subjects, quad_subjects]

    sentimentAnalyzer = SentimentIntensityAnalyzer()
    sentimentDict = {}
    for candidate,subjectsList in subjectsDict.items():
        candidateDf = dfDict[candidate]
        bi_sentiments = {}
        tri_sentiments = {}
        quad_sentiments = {}
        for subjects in subjectsList:
            print("Calculating sentiments for candidate %s" % candidate)
            for subject in subjects:
                match len(subject[0]): #Liší pro bigramy, trigramy a quadgramy
                    case 2:
                        bi_subject = subject[0]
                        bi_tweets = candidateDf[candidateDf.text.str.contains(r'^(?=.*%s)(?=.*%s)' % (bi_subject[0], bi_subject[1]))] #Získává z dataframu kandidáta pouze ty tweety, které obsahují daný ngram
                        subject_string = ' '.join(subject[0])
                        for tweet in bi_tweets.text:
                            bi_sentiments[subject_string] = bi_sentiments.get(subject_string,0) + sentimentAnalyzer.polarity_scores(tweet)['compound'] #Sčítá compound hodnoty Vader klasifikátoru
                        bi_sentiments[subject_string] = bi_sentiments.get(subject_string,1) / (len(bi_tweets.text) if len(bi_tweets.text) != 0 else 1) #Počítá průměr, oštřené proti dělení nulou
                        continue
                    case 3: #Stejné jako pro bigramy, ale pozměněné pro 3 hodnoty
                        tri_subjects = subjects[0][0]
                        tri_tweets = candidateDf[candidateDf.text.str.contains(r'^(?=.*%s)(?=.*%s)(?=.*%s)' % (tri_subjects[0], tri_subjects[1], tri_subjects[2]))]
                        subject_string = ' '.join(subject[0])
                        for tweet in tri_tweets.text:
                            tri_sentiments[subject_string] = tri_sentiments.get(subject_string, 0) + \
                                                            sentimentAnalyzer.polarity_scores(tweet)['compound']
                        tri_sentiments[subject_string] = tri_sentiments.get(subject_string, 1) / (len(
                            tri_tweets.text) if len(tri_tweets.text) != 0 else 1)
                        continue
                # Stejné jako pro bigramy, ale pozměněné pro 4 hodnoty
                quad_subjects = subjects[0][0]
                quad_tweets = candidateDf[candidateDf.text.str.contains(r'^(?=.*%s)(?=.*%s)(?=.*%s)(?=.*%s)' % (quad_subjects[0], quad_subjects[1], quad_subjects[2], quad_subjects[3]))]
                subject_string = ' '.join(subject[0])
                for tweet in quad_tweets.text:
                    quad_sentiments[subject_string] = quad_sentiments.get(subject_string, 0) + \
                                                     sentimentAnalyzer.polarity_scores(tweet)['compound']
                quad_sentiments[subject_string] = quad_sentiments.get(subject_string, 1) / (len(
                    quad_tweets.text) if len(quad_tweets.text) != 0 else 1)
        sentimentDict[candidate] = [bi_sentiments,tri_sentiments,quad_sentiments]
    for candidate in sentimentDict: #Vypisuje různé oblásti zájmu kandidátů a jejich oblíbenost
        print(candidate)
        print('Bigramy')
        print(sentimentDict[candidate][0])
        print('Trigramy')
        print(sentimentDict[candidate][1])
        print('Quadgramy')
        print(sentimentDict[candidate][2])







if __name__ == '__main__':
    main()

