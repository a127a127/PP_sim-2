
class Packet(object):
    def __init__(self, source, destination, data, pro_event_idx):
        self.source = source
        self.destination = destination
        self.data = data

        self.pro_event_idx = pro_event_idx
        
    def __str__(self):
        return str(self.__dict__)