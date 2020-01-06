from HardwareMetaData import HardwareMetaData
import collections

class CU(object):
    def __init__(self, cu_pos):
        self.position = cu_pos

        self.state = False

        ### event ready pool
        self.edram_rd_ir_erp = collections.deque()

        ### CU operation
        self.finish_cycle = 0
        self.cu_op_event = 0

        ### trigger event list
        self.pe_saa_trigger = []
        
        ### bottleneck analysis
        self.pure_idle_time = 0
        self.wait_transfer_time = 0
        self.wait_resource_time = 0
        self.pure_computation_time = 0

    def __str__(self):
        return str(self.__dict__)