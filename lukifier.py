from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag

lemmatizer = WordNetLemmatizer()


class Lukifier:
    def __init__(self, text):
        self.text = text
        self.tag = ""
        self.score = 0
        self.words = []
        self.polarity = self.swn_polarity()
        self.classification = self.classify()

    def penn_to_wn(self):
        """
        Convert between the PennTreebank tags to simple Wordnet tags
        """
        if self.tag.startswith('J'):
            return wn.ADJ
        elif self.tag.startswith('N'):
            return wn.NOUN
        elif self.tag.startswith('R'):
            return wn.ADV
        elif self.tag.startswith('V'):
            return wn.VERB
        elif self.tag.startswith('I'):
            return wn.VERB
        return None

    def swn_polarity(self):
        """
        Return a sentiment polarity: -1 = negative, 0 = neutral, 1 = positive
        """

        sentiment = 0.0
        tokens_count = 0

        raw_sentences = sent_tokenize(self.text)
        for raw_sentence in raw_sentences:
            tagged_sentence = pos_tag(word_tokenize(raw_sentence))

            negated = False
            for word, tag in tagged_sentence:
                self.tag = tag

                if word.endswith("n't") or word == "not":
                    negated = True

                if len(word) <= 2:
                    continue

                wn_tag = self.penn_to_wn()

                if wn_tag:
                    lemma = lemmatizer.lemmatize(word, pos=wn_tag)
                else:
                    continue

                if not lemma:
                    continue

                synsets = wn.synsets(word, pos=wn_tag)
                if not synsets:
                    synsets = wn.synsets(word)
                    if not synsets:
                        continue

                # Take the first sense, the most common
                synset = synsets[0]
                swn_synset = swn.senti_synset(synset.name())

                pos_score = swn_synset.pos_score()
                neg_score = swn_synset.neg_score()

                if negated:
                    pos_score = pos_score*-1
                    neg_score = neg_score*-1

                if neg_score == pos_score:
                    pos_score = 0

                diff = pos_score - neg_score
                self.words.append((word, diff))

                sentiment += diff
                tokens_count += 1

        # judgment call ? Default to positive or negative
        if not tokens_count:
            return 0

        self.score = sentiment

        # positive sentiment
        if sentiment >= 0.125:
            return 1

        # negative sentiment
        elif sentiment <= -0.125:
            return -1

        # neutral sentiment
        return 0

    def classify(self):
        if self.polarity == 1:
            return "POSITIVE"
        elif self.polarity == -1:
            return "NEGATIVE"
        return "NEUTRAL"
