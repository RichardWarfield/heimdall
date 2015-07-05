
def f(a,b,x):
    x = a
    return x*b*x

f(a,b,x=1)


def f2(a,b,c,d):
    """ Challenge for placing guards: end of matrix mult optimization ends in another fn."""
    rng = np.random.RandomState(55)
    # Start
    j = rng.uniform(size=(a,b))
    k = rng.uniform(size=(b,c))
    m = rng.uniform(size=(c,d))
    tmp1 = np.dot(j,k)
    y = h(tmp1,m)
    return y

def h(a,b):
    # End
    return np.dot(a,b)

def g2():
    if True:
        print "Hello"
    else:
        print "Impossible!"

def f2(a,b,c,d):
    """ Challenge for placing guards: start and end in different indentation levels.
    The whole If could be wrapped in guards, but what if there are assumptions..."""
    rng = np.random.RandomState(55)
    if True:
        rowsize = a+rng.randint(0,5)
        # First statement. Note that if we wrapped the whole If in guards then we
        # would be in trouble because rowsize is not known until now.
        j = rng.uniform(size=(rowsize,b))
        k = rng.uniform(size=(b,c))

    m = rng.uniform(size=(c,d))
    tmp1 = np.dot(j,k)
    y = np.dot(tmp1,m)
    # End
    return y
