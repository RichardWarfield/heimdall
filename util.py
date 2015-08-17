import astroid

builder = astroid.builder.AstroidBuilder()

def mainvar(name, val):
    import __main__
    __main__.__dict__[name] = val



def partition(lst, key_fn):
    """ Utility function to separate a list into a dictionary of lists, separated by
    key equivalence groups """
    res = {}
    for x in lst:
        key = key_fn(x)
        if key in res:
            res[key].append(x)
        else:
            res[key] = [x]
    return res

