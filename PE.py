from OnChipBuffer import OnChipBuffer
from HardwareMetaData import HardwareMetaData
from CU import CU
import collections

class PE(object):
    def __init__(self, pe_pos, input_bit):
        self.position = pe_pos

        self.state = False

        ### for edram read event 
        #self.state_edram_rd_ir = False
        self.idle_eventQueuing_CU = collections.deque() # CU為idle且有event正在queuing的position
        self.eventQueuing_CU = [] # for perfomance analysis
        self.edram_rd_event = None
        self.edram_rd_cu_idx = None
        self.edram_rd_cycle_ctr = 0
        self.data_to_ir_ing = False

        ### for cu operation
        self.cu_op_list = []

        self.edram_buffer = OnChipBuffer(input_bit)
        self.edram_buffer_i = OnChipBuffer(input_bit)
        self.input_require = []
        
        ### events per cycle
        # 要在一個cycle完成最多可能來自所有CU的資料
        self.pe_saa_epc = HardwareMetaData().Router_num * HardwareMetaData().PE_num * HardwareMetaData().CU_num 
        self.activation_epc = self.pe_saa_epc

        ### event ready pool
        self.pe_saa_erp = []
        self.activation_erp = collections.deque()
        self.edram_wr_erp   = collections.deque()
        self.edram_rd_pool_erp = []

        ### trigger event
        self.cu_op_trigger = 0
        self.activation_trigger = []
        self.edram_wr_trigger = []
        self.edram_rd_pool_trigger = []
        self.edram_rd_ir_trigger = []
        self.pe_saa_trigger = [] # for data transfer

        ### generate CU
        self.CU_array = []
        self.gen_cu()

        ### performance analysis
        self.pure_idle_time = 0
        self.wait_transfer_time = 0
        self.wait_resource_time = 0
        self.pure_computation_time = 0

        ### Energy
        self.Edram_buffer_energy     = 0.
        self.Bus_energy              = 0.
        self.PE_shift_and_add_energy = 0.
        self.Or_energy               = 0.
        self.Activation_energy       = 0.
        self.Pooling_energy          = 0.

        self.CU_shift_and_add_energy = 0.
        self.CU_dac_energy           = 0.
        self.CU_adc_energy           = 0.
        self.CU_crossbar_energy      = 0.
        self.CU_IR_energy            = 0.
        self.CU_OR_energy            = 0.

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