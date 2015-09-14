import util
from util import *
import tempfile
import astroid
import modcode
from modcode import make_astroid_node
import Cython.Compiler.Main as cymain
import pyximport.pyximport as pyxi
import numpy as np

import logging
logger = logging.getLogger(__name__)

def loop_to_cython(dfg, loopnode, var_types):
    try:
        in_edges = dfg.get_incoming_edges(loopnode)
        out_edges = dfg.get_outgoing_edges(loopnode)

        funname = util.unique_var("newfunc")

        argnames = [e.label for e in in_edges] + [v[0] for v in loopnode.child_dfg.external_deps]
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

        typed_args = ','.join([type_to_ctype(var_types.get(a, None))+' '+a for a in argnames])
        typedefs = ''
        code =    '    ' + ('\n    '.join([x.as_string() for x in loopnode.ast_node.body]))

        pyx_str = """
from __future__ import division
import numpy as np
cimport numpy as np

def {funname}({typed_args}):
    {typedefs}
{code}
    {retnode}
    """.format(funname=funname, typed_args=typed_args, typedefs=typedefs,
            code=modcode.indent(loopnode.ast_node.as_string(), 4), retnode=retnode.as_string())

        # TODO
        print "code is", pyx_str

        pyxfile = tempfile.NamedTemporaryFile(suffix='.pyx', delete=False)
        pyxfile.file.write(pyx_str)
        pyxfile.close()
        print "created cython file ", pyxfile.name
        #compiled_mod = cymain.compile(pyxfile.name)
        modname = unique_var('newmodule')
        mod = pyxi.load_module(modname, pyxfile.name)
        return getattr(mod, funname), argnames, retnames

    except:
        logger.exception("Oh crap! in loop_to_cython")
        raise


def type_to_ctype(typeinfo):
    if typeinfo is None:
        return ''
    typ, dtype, shape = typeinfo

    if issubclass(typ, np.ndarray):
        if dtype is not None and shape is not None:
            return 'np.ndarray[np.%s_t, ndim=%i]' % (str(dtype), len(shape))

    typdict = {int: 'int', np.ndarray: 'np.ndarray', float: 'float' }
    return typdict.get(typ, '')
