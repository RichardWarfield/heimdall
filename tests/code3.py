import numpy as np
import time

def g(n):
    for i in range(n):
        e = 2500
        t = time.time()
        x = f1(1500, e, 3000, 1000)
        print "g iter %i Took %f seconds, sum is %f" %(i, time.time() - t, x.sum())
    print x.shape

def func2(a, b):
    return np.dot(a, b)

def func1(a,b,c):
    return func2(a, np.dot(b, c))


def f1(a,b,c,d):
    rng = np.random.RandomState(55)
    j = rng.uniform(size=(a,b))
    k = rng.uniform(size=(b,c))
    m = rng.uniform(size=(c,d))
    return func1(j,k,m)
