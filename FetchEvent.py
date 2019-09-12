from HardwareMetaData import HardwareMetaData
class FetchEvent():
    def __init__(self, event):
        self.fetch_cycle = HardwareMetaData().Fetch_cycle
        self.cycles_counter = 0
        self.event = event
        
    def __str__(self):
        return str(self.__dict__)