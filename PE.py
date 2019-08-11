from OnChipBuffer import OnChipBuffer
from HardwareMetaData import HardwareMetaData
from CU import CU

class PE(object):
    def __init__(self, pe_pos, input_bit):
        ### hardware state flags
        self.position = pe_pos

        ### for buffer size analysis
        self.edram_buffer = OnChipBuffer(input_bit)
        self.input_require = []

        ### for mapping
        self.Pooling = []
        
        ### events per cycle
        self.pe_saa_epc = HardwareMetaData().CU_num * HardwareMetaData().Xbar_w * input_bit
        self.activation_epc = 16
        self.edram_wr_epc = 16 # 一個cycle可寫幾筆資料
        self.pooling_epc = 16

        ### event ready pool
        self.pe_saa_erp = []
        self.activation_erp = []
        #self.edram_rd_ir_erp = []
        self.edram_rd_pool_erp = []
        self.pooling_erp = []
        self.edram_wr_erp = []
        

        ### trigger event list
        self.activation_trigger = []
        self.edram_wr_trigger = []
        self.edram_rd_pool_trigger = []
        self.edram_rd_ir_trigger = []
        self.pooling_trigger = []
        
        self.pe_saa_trigger = [] # for data transfer

        ### generate CU
        self.CU_array = []
        self.gen_cu()

        self.reset()
    
    def reset(self):
        ### state
        self.state_edram_rd_pool = False
        self.state_pe_saa = [False] * self.pe_saa_epc
        self.state_activation = [False] * self.activation_epc
        self.state_edram_wr = [False] * self.edram_wr_epc
        self.state_pooling = [False] * self.pooling_epc

    def gen_cu(self):
        rty, rtx, pey, pex = self.position[0], self.position[1], self.position[2], self.position[3] 
        CU_num_y = HardwareMetaData().CU_num_y
        CU_num_x = HardwareMetaData().CU_num_x
        for cuy in range(CU_num_y):
            for cux in range(CU_num_x):
                cu_pos = (rty, rtx, pey, pex, cuy, cux)
                self.CU_array.append(CU(cu_pos))

    def check_state(self):
        if self.state_edram_rd_pool or True in self.state_pe_saa or True in self.state_activation:
            return True
        elif True in self.state_edram_wr or True in self.state_pooling:
            return True
        for cu in self.CU_array:
            if cu.check_state():
                return True
        return False

    def __str__(self):
        return str(self.__dict__)