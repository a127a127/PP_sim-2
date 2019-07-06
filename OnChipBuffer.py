from HardwareMetaData import HardwareMetaData

class OnChipBuffer():
    def __init__(self, input_bit):
        self.eDRAM_buffer_size = HardwareMetaData().eDRAM_buffer_size # KB
        self.eDRAM_buffer_bank_num = HardwareMetaData().eDRAM_buffer_bank_num
        self.eDRAM_buffer_bus_width = HardwareMetaData().eDRAM_buffer_bus_width # bit
        
        self.num_of_data = self.eDRAM_buffer_size * 1024 * 8 / input_bit

        self.buffer = []

    def check(self, data): 
        for d in self.buffer:
            if d == data:
                return True
        return False
    
    def put(self, data): # put a data to buffer
        if data in self.buffer:
            ### TODO: FIFO switch
            return
        if len(self.buffer) < self.num_of_data:
            self.buffer.append(data)
        else: # FIFO
            del self.buffer[0]
            self.buffer.append(data)
        return

    def get(self):
        if not self.empty():
            d = self.buffer[0]
            del self.buffer[0]
            return d
        else:
            print('buffer is empty..')
            return False

    def empty(self):
        if len(self.buffer) == 0:
            return True
        else:
            return False

    def __str__(self):
        return str(self.__dict__)