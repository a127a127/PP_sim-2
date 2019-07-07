import math
class TransferEvent():
    def __init__(self, event, src, des, des_cu):
        self.event = event
        self.source_pe = src
        self.destination_pe = des
        self.destination_cu = des_cu

        self.transfer_cycle = self.compute_cycle() 
        self.cycles_counter = 0
    
    def compute_cycle(self):
        ## Todo: auto compute cycle
        src_rty,  src_rtx = self.source_pe[0], self.source_pe[1]
        des_rty,  des_rtx = self.destination_pe[0], self.destination_pe[1]
        distance = abs(src_rty - des_rty) + abs(src_rtx - des_rtx)
        # traval_time = distance * router_transfer_time
        # traval_cycle = math.ceil(traval_time/cycle_time)
        traval_cycle = distance * 2 + 4 #  目前先假設router之間傳輸要2個cycle 4: pe_to_rt rt_to_pe
        return traval_cycle

    def __str__(self):
        return str(self.__dict__)