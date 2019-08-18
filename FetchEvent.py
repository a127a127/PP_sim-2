class FetchEvent():
    def __init__(self, event, index):
        self.fetch_cycle = 10
        self.cycles_counter = 0
        self.event = event
        self.index = index # [pe_idx, cu_idx]
        
    def __str__(self):
        return str(self.__dict__)