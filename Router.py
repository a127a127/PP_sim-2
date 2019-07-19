class Router(object):
    def __init__(self, h, w):
        self.pos = (h, w)
        
        self.queue_size = 20 # n packets 

        # input port
        self.arrived_order = []
        self.input_queue = {"pe_north_west":[], \
                            "pe_north_east":[], \
                            "pe_south_west":[], \
                            "pe_south_east":[], \
                            "east":[], \
                            "west":[], \
                            "south":[], \
                            "north":[]
                            }

    def packet_in(self, packet, port_type):
        if self.queue_size == len(self.input_queue[port_type]):
            # queue full
            print("router queue full")
            exit()
            return False
        else:
            self.arrived_order.append(port_type)
            self.input_queue[port_type].append(packet)
            return True
    
    def step(self):
        port_type = self.arrived_order[0]
        del self.arrived_order[0]
        packet = self.input_queue[port_type][0]
        del self.input_queue[port_type][0]
        
        # X-Y routing
        if packet.destination[1] > self.pos[1]:
            print("\t", packet.data, "往東走")
            return packet, "east"
        elif packet.destination[1] < self.pos[1]:
            print("\t", packet.data, "往西走")
            return packet, "west"
        else:
            if packet.destination[0] > self.pos[0]:
                print("\t", packet.data, "往南走")
                return packet, "south"
            elif packet.destination[0] < self.pos[0]:
                print("\t", packet.data, "往北走")
                return packet, "north"
            else:
                #抵達router
                print("\t", packet.data, "抵達router", end="")
                if packet.destination[2] == 0 and packet.destination[3] == 0:
                    print("西北")
                    return packet, "pe_north_west"
                elif packet.destination[2] == 0 and packet.destination[3] == 1:
                    print("東北")
                    return packet, "pe_north_east"
                elif packet.destination[2] == 1 and packet.destination[3] == 0:
                    print("西南")
                    return packet, "pe_south_west"
                else:  #packet.destination[2] == 1 and packet.destination[3] == 1:
                    print("東南")
                    return packet, "pe_south_east"
                
    def __str__(self):
        return str(self.__dict__)

