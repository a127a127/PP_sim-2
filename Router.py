import collections

class Router(object):
    def __init__(self, h, w):
        self.pos = (h, w)
        
        self.queue_size = 20000 # n packets  large enough

        # input port
        self.arrived_order = collections.deque() # self.arrived_order = list()
        self.input_queue = {"off_chip": collections.deque(),
                            "pe_north_west": collections.deque(),
                            "pe_north_east": collections.deque(),
                            "pe_south_west": collections.deque(),
                            "pe_south_east": collections.deque(),
                            "east":  collections.deque(),
                            "west":  collections.deque(),
                            "south": collections.deque(),
                            "north": collections.deque()
                            }

    def packet_in(self, packet, port_type):
        # if self.queue_size == len(self.input_queue[port_type]):
        #     # TODO: deal with queue full 
        #     # queue full
        #     print("router queue full")
        #     exit()
        #     return False
        # else:
        #     self.arrived_order.append(port_type)
        #     self.input_queue[port_type].append(packet)
        #     return True
        self.arrived_order.append(port_type)
        self.input_queue[port_type].append(packet)
        return True
    
    def step(self):
        #port_type = self.arrived_order[0]
        #del self.arrived_order[0]
        port_type = self.arrived_order.popleft()
        packet =  self.input_queue[port_type].popleft() 
        #packet =self.input_queue[port_type][0]
        #del self.input_queue[port_type][0]
        
        # X-Y routing
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
                #print("\t", packet.data, "arrive router", end="")
                # if packet.destination[2] == 0 and packet.destination[3] == 0:
                #     return packet, "pe_north_west"
                # elif packet.destination[2] == 0 and packet.destination[3] == 1:
                #     return packet, "pe_north_east"
                # elif packet.destination[2] == 1 and packet.destination[3] == 0:
                #     return packet, "pe_south_west"
                # elif packet.destination[2] == 1 and packet.destination[3] == 1:
                #     return packet, "pe_south_east"
                # else:
                #     print("router traverse error")
                #     exit()

    def __str__(self):
        return str(self.__dict__)

