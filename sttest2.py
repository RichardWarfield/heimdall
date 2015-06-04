import heimdall
import sys, os
import numpy as np

reload(heimdall)
watch = heimdall.Watcher()

def f(a,b):
    c = np.random.uniform(size=(a,b))
    d = np.random.uniform(size=(b,a))
    np.dot(d,c)

    return np.einsum('ij,jk->ik', c,d) + 66

def g():
    for i in range(3):
        f(1000,1000)


watch.run('g()', globals(), locals())

import pprint
pprint.pprint(watch.trace)
pprint.pprint(watch.cum_times)
pprint.pprint(watch.shitlist)
