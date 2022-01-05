import sys
import time 


class Logger:
    def __init__(self, outfile):
        self.terminal = sys.stdout 
        self.log = open(outfile, 'w')
        sys.stdout = self 
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
class Timer:
    def __init__(self, starting_msg=None):
        self.start = time.time()
        self.stage_start = self.start 
        
        if starting_msg is not None:
            print(starting_msg, time.ctime(time.time()))
            
    def update_progress(self, progress):
        self.elapsed = time.time() - self.start
        self.est_total = self.elapsed / progress 
        self.est_remaining = self.est_total - self.elapsed 
        self.est_finish = int(self.start + self.est_total)

    def str_est_finish(self):
        return str(time.ctime(self.est_finish))
    
    def get_stage_elapsed(self):
        return time.time() - self.stage_start
    
    def reset_stage(self):
        self.stage_start = time.time()
        
        
