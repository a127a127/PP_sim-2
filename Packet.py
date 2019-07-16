
class Packet(object):
    def __init__(self, source, destination, data):
        self.source = source
        self.destination = destination
        self.data = data