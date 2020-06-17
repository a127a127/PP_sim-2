class FetchEvent():
    __slots__ = ["event", "des_pe", "data"]
    def __init__(self, event, des_pe, data):
        self.event = event
        self.des_pe = des_pe
        self.data  = data
        
    def __str__(self):
        return str({"fetch_cycle:": self.fetch_cycle, "cycles_counter:": self.cycles_counter, "event": self.event, "data:": self.data})
        