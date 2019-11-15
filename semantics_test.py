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
        "Hello, how are you?" : "POSITIVE",
        "Goodbye forever" : "NEGATIVE"
    }
    for sentence in sentences:
        classification = Lukifier(sentence).classification
        if classification != sentences[sentence]:
            print(
                "Sentence: {} | Expected {} | Actual {}".format(
                    sentence, sentences[sentence], classification
                )
            )


test()
