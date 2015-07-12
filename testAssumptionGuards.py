from optimizer import guards_between
import astroid, optimizer

def f1(a,b,c,d):
    rng = np.random.RandomState(55)
    j = rng.uniform(size=(a,b))
    k = rng.uniform(size=(b,c))
    m = rng.uniform(size=(c,d))
    tmp1 = np.dot(j,k)
    y = np.dot(tmp1,m)
    return y

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

def f3(a,b,c,d):
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

def f4(a,b,c,d):
    if True:
        rowsize = a+rng.randint(0,5)
        # First statement. Note that if we wrapped the whole If in guards then we
        # would be in trouble because rowsize is not known until now.
        j = rng.uniform(size=(rowsize,b))
        k = rng.uniform(size=(b,c))

    if True:
        m = rng.uniform(size=(c,d))
        tmp1 = np.dot(j,k)
        y = np.dot(tmp1,m)
    # End
    return y

def test1():
    ast = astroid.MANAGER.ast_from_file(__file__)
    print __file__
    print ast.body[2] # f1
    guards = guards_between(ast.body[2], 0, 10)
    assert len(guards)==1
    fn, startidx, endidx = guards[0]
    assert startidx == 0 and endidx==5

def test_insert_guards():
    ast = astroid.MANAGER.ast_from_file(__file__)
    fn = ast.body[2] # f1
    print fn
    optimizer.insert_guards('ass_ok', fn, 7, 8)


def test_replace_subgraph_and_code():
    import data_flow
    ast = astroid.MANAGER.ast_from_file(__file__)

    stmts = [
           (__file__, 5, 'line'),
           (__file__, 6, 'line'),
           (__file__, 7, 'line'),
           (__file__, 8, 'line'),
           (__file__, 9, 'line'),
           (__file__, 10, 'line'),
           (__file__, 11, 'return')
    ]

    dfg = data_flow.analyze_flow(stmts)

    #nodes = [n for n in dfg.nodes if n.stmt_idx ==4 or (n.stmt_idx==5 and "AssName(y)" not in str(n))]
    #nodes += [n for n in dfg.nodes if dfg.get_outgoing_edges(n) and list(dfg.get_outgoing_edges(n))[0].label in ('m','k', 'j')]
    nodes,edges = dfg.subgraph_between([n for n in dfg.nodes if 'Call rng.uniform' in str(n)],
            [n for n in dfg.nodes if "10(Call np.dot" in str(n)][0])
    print "nodes", nodes
    #edges = [e for e in dfg.edges if e.n1 in nodes and e.n2 in nodes]
    in_edges = [e for e in dfg.edges if e.n2 in nodes and e.n1 not in nodes]
    out_edges = [e for e in dfg.edges if e.n1 in nodes and e.n2 not in nodes]
    new_expr = 'np.dot(j, np.dot(k,m))'
    ass_expr = [n for n in dfg.nodes if 'Call rng.uniform' in str(n)][0]
    assumptions = {ass_expr: '{1}'}
    dfd = dfg.draw_digraph(colors={n: 'red' for n in nodes})
    print dfd.view()

    optimizer.replace_subgraph_and_code(dfg, nodes, edges, in_edges, out_edges, new_expr, assumptions)

