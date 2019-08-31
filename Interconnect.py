from Router import Router
import numpy as np
from HardwareMetaData import HardwareMetaData

class Interconnect(object):
    def __init__(self, rt_h, rt_w):
        self.router_array = []
        for h in range(rt_h):
            for w in range(rt_w):
                self.router_array.append(Router(h, w))

        self.router_array = np.array(self.router_array)
        self.router_array = self.router_array.reshape(rt_h, rt_w)
        self.step_energy_consumption = 0
        self.router_step_energy = HardwareMetaData().Energy_router

        self.arrived_list = []

        self.packet_in_module_ctr = 0

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

    def step(self):
        self.step_energy_consumption = 0
        packet_transfer = []
        for h in range(self.router_array.shape[0]):
            for w in range(self.router_array.shape[1]):
                if not self.router_array[h][w].arrived_order:
                    continue
                packet, port_type = self.router_array[h][w].step()
                self.step_energy_consumption += self.router_step_energy
                if port_type == "east":
                    packet_transfer.append((h, w+1, packet, port_type))
                elif port_type == "west":
                    packet_transfer.append((h, w-1, packet, port_type))
                elif port_type == "south":
                    packet_transfer.append((h+1, w, packet, port_type))
                elif port_type == "north":
                    packet_transfer.append((h-1, w, packet, port_type))
                else:
                    # arrived
                    self.arrived_list.append(packet)
        
        for p in packet_transfer:
            h = p[0]
            w = p[1]
            packet = p[2]
            port_type = p[3]
            self.router_array[h][w].packet_in(packet, port_type)

    def get_arrived_packet(self):
        A = self.arrived_list.copy()
        self.packet_in_module_ctr -= len(A)
        self.arrived_list = []
        return A

    def busy(self):
        if self.packet_in_module_ctr == 0:
            return False
        else:
            return True

    def __str__(self):
        return str(self.__dict__)
    