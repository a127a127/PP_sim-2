from HardwareMetaData import HardwareMetaData
class FetchEvent():
    __slots__ = ["fetch_cycle", "cycles_counter", "event", "data"]
    def __init__(self, event, data):
        self.fetch_cycle = HardwareMetaData().Fetch_cycle
        self.cycles_counter = 0
        self.event = event
        self.data  = data
        
    def __str__(self):
        return str({"fetch_cycle:": self.fetch_cycle, "cycles_counter:": self.cycles_counter, "event": self.event, "data:": self.data})
        