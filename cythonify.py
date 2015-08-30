import tempfile
import astroid
import modcode
from modcode import make_astroid_node
import Cython.Compiler.Main as cymain
import pyximport.pyximport as pyxi

def loop_to_cython(dfg, loopnode):
    in_edges = dfg.get_incoming_edges(loopnode)
    out_edges = dfg.get_outgoing_edges(loopnode)

    funname = "newfunc"

    argnames = [e.label for e in in_edges] + [v[0] for v in loopnode.dfg.external_deps]
    args=[make_astroid_node(astroid.AssName, name=a) for a in argnames]

    retnames = [e.label for e in out_edges]

    retnode = make_astroid_node(astroid.Return,
            value=make_astroid_node(astroid.Tuple,
                elts=[make_astroid_node(astroid.Name, name=r) for r in retnames])
            )
    newfunc = modcode.make_astroid_function(funname, "docs go here",
            body=[loopnode.ast_node, retnode],
            args=make_astroid_node(astroid.Arguments,
                args=args, defaults=[], kwonlyargs=[], kw_defaults=[])
        )


    # TODO
    print "argnames is ", argnames
    code = "import numpy as np\n"+newfunc.as_string()
    print "code is", code

    pyxfile = tempfile.NamedTemporaryFile(suffix='.pyx', delete=False)
    pyxfile.file.write(code)
    pyxfile.close()
    print "created cython file ", pyxfile.name
    #compiled_mod = cymain.compile(pyxfile.name)
    mod = pyxi.load_module('crazyloopingmodule', pyxfile.name)
    return getattr(mod, funname), argnames, retnames


