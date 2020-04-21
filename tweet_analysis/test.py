from random import randint

def get_range(size, orig_size):
    length = int(size*orig_size)
    start = randint(0, orig_size)
    if (start + length) > orig_size:
        start = (start + length) % orig_size
    end = start + length
    return start, end


def main(X):
    orig_size = len(X)
    sizes = [i*0.125 for i in range(1, 9)]
    lengths = [int(size*orig_size) for size in sizes]
    print("Training on tweets of sizes: {}".format(lengths))
    print("Original size is: {}".format(orig_size))
    for size in sizes:
        start, end = get_range(size, orig_size)
        print(start, end, end-start)

X = [randint(0, 46208) for i in range(46208)]

main(X)