import heimdall
heimdall = reload(heimdall)
import data_flow; data_flow = reload(data_flow)
import watcher; watcher = reload(watcher)
import optimizer; optimizer = reload(optimizer)
from tests import code2
import asizeof, gc

heim = heimdall.Heimdall(max_nesting=2, prof_shot_time=4)
heim.run('code2.g(5)', globals(), locals())


sizes = []
print "number of objects is ", len(gc.get_objects())
biggest, biggest_size = None, 0
for (i,o) in enumerate(gc.get_objects()):
    if i%100 ==0:
        print i
    try:
        size= asizeof.asizeof(o)
        if size > 1000000:
            print o,size
            sizes.append((o,size ))
        if size > biggest_size:
            print "new biggest", size, o
            biggest, biggest_size = o, size
    except:
        pass
