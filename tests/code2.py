import numpy as np
import time

def g(n):
    for i in range(n):
        e = 2500
        t = time.time()
        x = f1(1400, e, 2800, 900)
        print "g iter %i Took %f seconds, sum is %f" %(i, time.time() - t, x.sum())
    print x.shape

def jfunc(rng, a,b):
    return rng.uniform(size=(a,b))

def kfunc(rng, b,c):
    return rng.uniform(size=(b,c))

def f1(a,b,c,d):
    rng = np.random.RandomState(55)
    m = rng.uniform(size=(c,d))
    j = jfunc(rng, a,b)
    k = kfunc(rng, b,c)
    tmp1 = np.dot(j,k)
    y = np.dot(tmp1,m)
    return y
