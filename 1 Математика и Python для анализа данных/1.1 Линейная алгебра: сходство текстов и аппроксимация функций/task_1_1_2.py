import numpy as np
from scipy.linalg import solve


def f(x):
    return np.sin(x/5) * np.exp(x/10) + 5 * np.exp(-x/2)


# многочлен первой степени

# x = 1:  w0 + 1*w1 = 3.252216865271419
# x = 15: w0 + 15*w1 = 0.6352214195786656

# многочлен второй степени

# x = 1:  w0 + 1*w1 + 1*w2 = 3.252216865271419
# x = 8:  w0 + 8*w1 + 64*w2 = 2.316170159053224
# x = 15: w0 + 15*w1 + 225*w2 = 0.6352214195786656

# многочлен третей степени

# x = 1:  w0 + 1*w1 + 1*w2 + 1*w3 = 3.252216865271419
# x = 4:  w0 + 4*w1 + 16*w2 + 64*w3 = 1.7468459495903677
# x = 10:  w0 + 10*w1 + 100*w2 + 1000*w3 = 2.316170159053224
# x = 15: w0 + 15*w1 + 225*w2 + 3375*w3 = 0.6352214195786656

X = np.array([[1, 1, 1, 1], [1, 4, 16, 64], [1, 10, 100, 1000], [1, 15, 225, 3375]])
y = np.array([f(1), f(4), f(10), f(15)])
W = solve(X, y)


if __name__ == '__main__':
    print(f(1), f(4), f(10), f(15))
    print(' '.join([str(w) for w in W]))

    with open('submission-2.txt', 'w') as file:
        file.write(' '.join([str(w) for w in W]))
