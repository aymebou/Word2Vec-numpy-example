from random import randint


def random_length_bound_sequence_around(index, array):
    low = index - randint(1, 5)
    up = index + randint(1, 5)
    out = []
    for k in range(low, up + 1):
        if 0 < k < len(array):
            out.append(array[k])
    return out
