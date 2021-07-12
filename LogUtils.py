import logging
import os
import time

current_path=os.path.dirname(__file__)
current_time=time.strftime('%Y-%m-%d-%H_%M_%S',time.localtime(time.time()))
filename=current_time+'.log'
logpath=os.path.join(current_path,'.','log/',filename)

class LogUtils(object):
    def __init__(self,loggername=None):
        self.log_path_name=logpath
        self.log=logging.getLogger(loggername)
        self.log.setLevel(logging.DEBUG)
        formatter=logging.Formatter('[%(asctime)s] - %(filename)s [line:%(lineno)d] - %(levelname)s - %(message)s')
        self.sh=logging.StreamHandler()
        self.sh.setLevel(logging.INFO)
        self.sh.setFormatter(formatter)

        self.fh=logging.FileHandler(logpath,'a',encoding='utf-8')
        self.fh.setLevel(logging.INFO)
        self.fh.setFormatter(formatter)

        self.log.addHandler(self.sh)
        self.log.addHandler(self.fh)

        self.sh.close()
        self.fh.close()

    def get_log(self):
        return self.log

# logger=LogUtils('testlog').get_log()
#
# if __name__=='__main__':
#     logger.info('test message!')