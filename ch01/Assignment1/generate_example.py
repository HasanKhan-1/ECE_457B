import random

def generate_example():
    dimlo = 10
    dimhi = 1000

    # Examples are within bigdim
    bigdimlo = -10
    bigdimhi = 1100

    n = 2 # num dimensions

    while True:
        example = list()
        for i in range(n):
            example.append(random.randint(bigdimlo, bigdimhi))

        # Is it positive or negative?
        ispos = True
        for d in example:
            if d < dimlo or d > dimhi:
                ispos = False
                break

        yield tuple((example, ispos))
