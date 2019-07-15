class NetworkTransfer(object):
    def __init__(self, one_step_energy):
        self.transfer_list = []
        self.one_step_energy = one_step_energy
        self.interconnect_energy_total = 0
        self.transfer_count = 0

    def step(self):
        arrived = []
        for TE in self.transfer_list.copy():
            self.transfer_count += 1
            TE_idx = self.transfer_list.index(TE)
            self.transfer_list[TE_idx].cycles_counter += 1
            self.interconnect_energy_total += self.one_step_energy
            if self.transfer_list[TE_idx].cycles_counter == self.transfer_list[TE_idx].transfer_cycle:
                self.transfer_list.remove(TE)
                arrived.append(TE)
        return arrived 

    def __str__(self):
        return str(self.__dict__)