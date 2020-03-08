from HardwareMetaData import HardwareMetaData
class FetchEvent():
    def __init__(self, event, data):
        self.fetch_cycle = HardwareMetaData().Fetch_cycle
        self.cycles_counter = 0
        self.event = event
        self.data  = data
        
    def __str__(self):
        return str(self.__dict__)