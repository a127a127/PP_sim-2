from OnChipBuffer import OnChipBuffer
from HardwareMetaData import HardwareMetaData as HW
from ModelConfig import ModelConfig
from CU import CU
import collections

class PE(object):
    def __init__(self, pe_pos):
        self.position = pe_pos
        self.plot_idx = pe_pos[3] + pe_pos[2]*HW().PE_num_x + pe_pos[1]*HW().PE_num + pe_pos[0]*HW().Router_num_x*HW().PE_num
        self.state = False

        size = HW().eDRAM_buffer_size * 1024 * 8 / ModelConfig().input_bit
        self.edram_buffer = OnChipBuffer(size)
        self.buffer_size_util = [[], []] # [cycle, size]

        ### # Event queue's CU index
        self.edram_rd_cu_idx     = collections.deque() # set()
        self.cu_operation_cu_idx = collections.deque()

        ### event queue
        self.edram_rd_erp     = collections.deque()
        self.edram_rd_ir_erp  = [] # 一個CU一個edram_rd     event queue
        self.cu_operation_erp = [] # 一個CU一個cu_operation event queue
        for i in range(HW().CU_num):
            self.edram_rd_ir_erp.append(collections.deque())
            self.cu_operation_erp.append(collections.deque())
        self.pe_saa_erp       = collections.deque()
        self.activation_erp   = collections.deque()
        self.edram_wr_erp     = collections.deque()
        self.pooling_erp      = collections.deque()
        
        ### Performance analysis
        self.is_wait_resource = False
        self.is_wait_transfer = False
        self.pure_idle_time        = 0
        self.wait_transfer_time    = 0
        self.wait_resource_time    = 0
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

        ### Performance analysis
        self.busy_cu       = 0
        self.busy_other    = 0
        self.idle_transfer = 0
        self.idle_other    = 0

    def __str__(self):
        return str(self.__dict__)
