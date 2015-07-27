import astroid

builder = astroid.builder.AstroidBuilder()

def mainvar(name, val):
    import __main__
    __main__.__dict__[name] = val
