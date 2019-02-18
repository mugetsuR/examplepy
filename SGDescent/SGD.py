import random as rnd

# Подобие стохастического градиентного спуска
# выборка, количество признаков, вектор ответов, параметр сглаживания,темп обучения, z- порог
def SGD(listOfObjs, n, y, lmbd, z,tmp = 0.1):

    w = []          # вектор весов
                    # вектор ответов
    f = []          # Fj = (Fj(Xi)) i=1..L j=1..n (j-й столбец в матрице объекты-признаки)

    for j in range(n):
        fj = []
        for i in listOfObjs:
            fj.append(i[j])
        f.append(fj)

                     # Wj = <y, Fj>/ <Fj, Fj> инициализация весов
    for j in range(n):
        wjc = 0
        wjd = 0

        for i in range(listOfObjs):
            wjc += y[i] * f[i][j]
            wjd += f[i][j] * f[i][j]

        w[j] = wjc / wjd
                    # оценка функционала
    def lossQ(pos = -1) :
        if pos == -1 :
            Q = 0
            for i in range(listOfObjs):
                L = 0
                for j in range(n):
                    L += w[j] * listOfObjs[i][j]
                L -= y[i]
                L *= L
                Q += L
            return Q
        else :
            L = 0
            for j in range(n):
                L += w[j] * listOfObjs[pos][j]
                L -= y[pos]
                L *= L
        return L

    def lossQGrad(i):
        grad = []
        for j in range(n):
            grad.append(listOfObjs[i][j])
        return grad
    Q = lossQ()
    while True:
        i = rnd.randint(0,len(listOfObjs))

        Li = lossQ(i)

        gradient = lossQGrad(i)
        for j in range(n):
            w[j]-=tmp*gradient[j]

        Q = (1 - lmbd)*Q - lmbd*Li
        if Q <z: return w