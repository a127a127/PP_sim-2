from XB import XB
from HardwareMetaData import HardwareMetaData
import collections

class CU(object):
    def __init__(self, cu_pos):
        self.position = cu_pos

        self.state = False

        self.edram_rd_event = None
        self.edram_rd_cycle_ctr = 0

        ### event ready pool
        self.edram_rd_ir_erp = collections.deque()
        self.cu_saa_erp = []

        ### CU operation
        self.finish_cycle = 0
        self.cu_op_event = 0

        ### trigger event list
        self.ou_trigger = [] # [pro_event, [cu_idx, xb_idx]]
        self.pe_saa_trigger = []
        self.cu_saa_trigger = []
        self.cu_op_trigger  = 0

        ### CU operation
        self.finish_cycle = 0
        
        ### generate XB
        self.XB_array = []
        self.gen_xb()
        
        self.state_edram_rd_ir = False
        
        ### bottleneck analysis
        self.pure_idle_time = 0
        self.wait_transfer_time = 0
        self.wait_resource_time = 0
        self.pure_computation_time = 0

    def reset(self):
        return

    def gen_xb(self):
        rty, rtx, pey, pex, cuy, cux = self.position[0], self.position[1], self.position[2], self.position[3] , self.position[4], self.position[5] 
        XB_num_y = HardwareMetaData().Xbar_num_y
        XB_num_x = HardwareMetaData().Xbar_num_x
        for xby in range(XB_num_y):
            for xbx in range(XB_num_x):
                xb_pos = (rty, rtx, pey, pex, cuy, cux, xby, xbx)
                self.XB_array.append(XB(xb_pos))
    
    def check_state(self):
        if self.state_edram_rd_ir or self.state:
            return True
        return False

    def __str__(self):
        return str(self.__dict__)