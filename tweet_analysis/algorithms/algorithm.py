class Algorithm:
    def __init__(self, data):
        self.name = data[0]
        self.accuracy = float(data[1])
        self.precision = float(data[2])
        self.recall = float(data[3])
        self.time = float(data[4])

    def __str__(self):
        return "{}:\nAcc - {}\nPre - {}\nRecall - {}\nTime - {}\n".format(
            self.name, self.accuracy, self.precision, self.recall, self.time)
