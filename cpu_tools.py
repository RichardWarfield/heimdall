
from __future__ import division

def get_jiffies():
    """ This is for Linux only"""

    with open("/proc/stat") as f:
        while(True):
            line = f.readline().rstrip()
            if line.startswith('cpu'):
                user, nice, system, idle, iowait, irq, softirq = [int(x) for x in line[5:].split(' ')][:7]
                return user, nice, system, idle, iowait, irq, softirq




class CpuMeter(object):
    def __init__(self):
        self.start()

    def start(self):
        all_jiffies = get_jiffies()
        self.start_total_jiffies = sum(all_jiffies)
        self.start_work_jiffies = sum(all_jiffies[:3])

    def finish(self):
        all_jiffies = get_jiffies()
        finish_total_jiffies = sum(all_jiffies)
        finish_work_jiffies = sum(all_jiffies[:3])

        try:
            return (finish_work_jiffies-self.start_work_jiffies) / float(finish_total_jiffies-self.start_total_jiffies)
        except ZeroDivisionError:
            # whatever
            return 1.0




