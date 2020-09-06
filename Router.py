import collections

class Router(object):
    def __init__(self, h, w):
        self.pos = (h, w)
        self.input_queue = collections.deque()
    
    def step(self):
        packet = self.input_queue.popleft()
        # X-Y
        if packet.destination[1] > self.pos[1]:
            return packet, "east"
        elif packet.destination[1] < self.pos[1]:
            return packet, "west"
        else:
            if packet.destination[0] > self.pos[0]:
                return packet, "south"
            elif packet.destination[0] < self.pos[0]:
                return packet, "north"
            else:
                return packet, "arrived"

    def __str__(self):
        return str(self.__dict__)

