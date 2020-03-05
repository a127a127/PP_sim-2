from Router import Router
import numpy as np
from HardwareMetaData import HardwareMetaData
import time

class Interconnect(object):
    def __init__(self, rt_h, rt_w, data_bit_width):
        self.router_array = []
        self.rt_h = rt_h
        self.rt_w = rt_w
        for h in range(rt_h):
            for w in range(rt_w):
                self.router_array.append(Router(h, w))

        self.router_array = np.array(self.router_array)
        self.router_array = self.router_array.reshape(rt_h, rt_w)
        self.step_energy_consumption = 0
        self.router_step_energy = HardwareMetaData().Energy_router * data_bit_width

        self.arrived_list = []

        self.packet_in_module_ctr = 0
        self.busy_router = set()

    def input_packet(self, packet):
        self.packet_in_module_ctr += 1
        source = packet.source

        rt_y, rt_x = source[0], source[1]
        pe_y, pe_x = source[2], source[3]
        
        if pe_y == -1 and pe_x == -1:
            port_type = "off_chip"
        elif pe_y == 0 and pe_x == 0:
            port_type = "pe_north_west"
        elif pe_y == 0 and pe_x == 1:
            port_type = "pe_north_east"
        elif pe_y == 1 and pe_x == 0:
            port_type = "pe_south_west"
        elif pe_y == 1 and pe_x == 1:
            port_type = "pe_south_east"
        else:
            print("type error")
            return
        self.router_array[rt_y][rt_x].packet_in(packet, port_type)
        self.busy_router.add((rt_y, rt_x))

    def step(self):
        self.step_energy_consumption = 0
        if self.packet_in_module_ctr == 0:
            return

        packet_transfer = []

        br = set()
        for rt_id in self.busy_router:
            h = rt_id[0]
            w = rt_id[1]
            port_type = self.router_array[h][w].arrived_order.popleft()
            packet =  self.router_array[h][w].input_queue[port_type].popleft()
            self.step_energy_consumption += self.router_step_energy
            if packet.destination[1] > w:
                packet_transfer.append((h, w+1, packet, "east"))
            elif packet.destination[1] < w:
                packet_transfer.append((h, w-1, packet, "west"))
            else:
                if packet.destination[0] > h:
                    packet_transfer.append((h+1, w, packet, "south"))
                elif packet.destination[0] < h:
                    packet_transfer.append((h-1, w, packet, "north"))
                else: # arrived
                    self.arrived_list.append(packet)
            if self.router_array[h][w].arrived_order:
                br.add((h, w))

        for p in packet_transfer:
            h = p[0]
            w = p[1]
            packet = p[2]
            port_type = p[3]
            self.router_array[h][w].packet_in(packet, port_type)
            br.add((h,w))

        self.busy_router = br

    def busy(self):
        if self.packet_in_module_ctr == 0:
            return False
        else:
            return True

    def __str__(self):
        return str(self.__dict__)
    