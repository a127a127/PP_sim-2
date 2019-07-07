class NetworkTransfer():
    def __init__(self):
        self.transfer_list = []

    def step(self):
        arrived = []
        for TE in self.transfer_list.copy():
            TE_idx = self.transfer_list.index(TE)
            self.transfer_list[TE_idx].cycles_counter += 1
            if self.transfer_list[TE_idx].cycles_counter == self.transfer_list[TE_idx].transfer_cycle:
                self.transfer_list.remove(TE)
                arrived.append(TE)
        return arrived 

    def __str__(self):
        return str(self.__dict__)