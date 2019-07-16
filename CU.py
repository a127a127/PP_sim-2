from XB import XB
from HardwareMetaData import HardwareMetaData

class CU(object):
    def __init__(self, cu_pos):
        self.position = cu_pos

        self.state = False # CU busy

        ### events per cycle
        ### Todo: 隨給定資料調整
        self.cu_saa_epc = 32

        ### event ready pool
        self.edram_rd_ir_erp = []
        self.cu_saa_erp = []

        ### trigger event list
        self.ou_operation_trigger = [] # [pro_event, [cu_idx, xb_idx]]
        self.pe_saa_trigger = []
        
        ### generate XB
        self.XB_array = []
        self.gen_xb()
        
        self.reset()
        
    def reset(self):
        # state
        self.state_edram_rd_ir = False
        self.state_cu_saa = [False] * self.cu_saa_epc

    def gen_xb(self):
        rty, rtx, pey, pex, cuy, cux = self.position[0], self.position[1], self.position[2], self.position[3] , self.position[4], self.position[5] 
        XB_num_y = HardwareMetaData().Xbar_num_y
        XB_num_x = HardwareMetaData().Xbar_num_x
        for xby in range(XB_num_y):
            for xbx in range(XB_num_x):
                xb_pos = (rty, rtx, pey, pex, cuy, cux, xby, xbx)
                self.XB_array.append(XB(xb_pos))

    def __str__(self):
        return str(self.__dict__)