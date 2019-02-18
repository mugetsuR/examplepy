import random as rnd

#Микромодификация SGD
def SAG(listOfObjs, n, y, lmbd, s, temp=0.1):
    w = []
    for j in range(n):
        w[j] = rnd.uniform(-1 / (2 * n), 1 / (2 * n))

    def lossQ(pos=-1):
        if pos == -1:
            Q = 0
            for i in range(listOfObjs):
                L = 0
                for j in range(n):
                    L += w[j] * listOfObjs[i][j]
                L -= y[i]
                L *= L
                Q += L
            return Q
        else:
            L = 0
            for j in range(n):
                L += w[j] * listOfObjs[pos][j]
                L -= y[pos]
                L *= L
        return L

    Q = lossQ()

    def lossQGrad(i):
        grad = []
        for j in range(n):
            grad.append(listOfObjs[i][j])
        return grad

    # Градиенты по всем X
    G = []
    for i in range(listOfObjs):
        G[i] = lossQGrad(i)

    while True:

        i = rnd.randint(0, len(listOfObjs))

        Li = lossQ(i)
        G[i] = lossQGrad(i)

        res = []
        for j in range(n):
            res[j] = 0
            for i in range(listOfObjs):
                res[j] += G[i][j]

        for j in range(n):
            w[j] -= temp * res[j]

        Q = (1 - lmbd) * Q - Q * Li

        if Q < s: return w
