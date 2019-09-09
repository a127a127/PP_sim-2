class IdleMetaData(object):
    def __init__(self, start_compute, finish_compute):
        self.start_compute = start_compute # first ou event execution
        self.finish_compute = finish_compute # edram wr 
        #self.finish_transfer = finish_transfer # 傳到目標buffer

    def __str__(self):
        return str(self.__dict__)