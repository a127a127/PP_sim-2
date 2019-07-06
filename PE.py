from OnChipBuffer import OnChipBuffer
from HardwareMetaData import HardwareMetaData
from CU import CU

class PE(object):
    def __init__(self, pe_pos, input_bit):
        ### hardware state flags
        #self.state_xbar = False
        #self.state_pooling = [False] * num_of_saa
        #self.state_activation = [False] * num_of_saa
        #self.state_saa = [False] * num_of_saa
        
        self.position = pe_pos

        ### for buffer size analysis
        self.edram_buffer = OnChipBuffer(input_bit)
        self.input_require = []

        ### for mapping
        self.Pooling = []
        
        ### events per cycle
        ###     Todo: 隨給定資料調整
        self.pe_saa_epc = 32
        self.activation_epc = 16
        self.edram_wr_epc = 16 # 一個cycle可寫幾筆資料
        self.pooling_epc = 16

        ### event ready pool
        self.pe_saa_erp = []
        self.activation_erp = []
        self.edram_rd_ir_erp = []
        self.edram_rd_pool_erp = []
        self.pooling_erp = []
        self.edram_wr_erp = []
        


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
        
        ### trigger event list
        self.activation_trigger = []
        self.edram_wr_trigger = []
        self.edram_rd_pool_trigger = []
        self.edram_rd_ir_trigger = []
        self.pooling_trigger = []

    def gen_cu(self):
        rty, rtx, pey, pex = self.position[0], self.position[1], self.position[2], self.position[3] 
        CU_num_y = HardwareMetaData().CU_num_y
        CU_num_x = HardwareMetaData().CU_num_x
        for cuy in range(CU_num_y):
            for cux in range(CU_num_x):
                cu_pos = (rty, rtx, pey, pex, cuy, cux)
                self.CU_array.append(CU(cu_pos))

    def __str__(self):
        return str(self.__dict__)