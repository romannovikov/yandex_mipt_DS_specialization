import numpy as np
from scipy.optimize import minimize, differential_evolution


bounds = [(1, 30)]


def f(x):
    return np.sin(x/5) * np.exp(x/10) + 5 * np.exp(-x/2)


def h(x):
    return int(f(x))


if __name__ == '__main__':

    # Задача №1 Минимизация гладкой функции

    result_1_2 = minimize(f, 2, method='BFGS')  # fun: 1.7452682903449388
    result_1_30 = minimize(f, 30, method='BFGS')  # fun: -11.898894665981285

    with open('submission-1.txt', 'w') as file:
        file.write(' '.join([str(round(result_1_2.fun, 2)), str(round(result_1_30.fun, 2))]))

    # Задача №2 Глобальная оптимизация

    result_2_ev = differential_evolution(f, bounds)  # fun: -11.898894665981285

    with open('submission-2.txt', 'w') as file:
        file.write(str(round(result_2_ev.fun.tolist()[0], 2)))

    # Задача №3 Минимизация негладкой функции

    result_3_30 = minimize(h, 30, method='BFGS')  # fun: -5
    result_3_ev = differential_evolution(h, bounds)  # fun: -11.0

    with open('submission-3.txt', 'w') as file:
        file.write(' '.join([str(result_3_30.fun), str(result_3_ev.fun)]))
