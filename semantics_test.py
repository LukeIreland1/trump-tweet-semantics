from lukifier import Lukifier


def test():
    sentences = {
        "I am sick of you": "NEGATIVE",
        "I am happy": "POSITIVE",
        "I am angry": "NEGATIVE",
        "I love you": "POSITIVE",
        "I hate you": "NEGATIVE",
        "I am furious": "NEGATIVE",
        "Go away, I don't like you": "NEGATIVE",
        "Make America Great Again": "POSITIVE",
        "Hello, how are you?": "POSITIVE",
        "Goodbye forever": "NEGATIVE"
    }
    output = "Sentence: {} | Expected {} | Actual {} | Score {} | Words {}"
    for sentence in sentences:
        classifier = Lukifier(sentence)
        classification = classifier.classification
        score = classifier.score
        if classification != sentences[sentence]:
            print(
                output.format(
                    sentence, sentences[sentence], classification, score,
                    classifier.words
                )
            )


test()
