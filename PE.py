from OnChipBuffer import OnChipBuffer
import collections

class PE(object):
    def __init__(self, hw_config, input_bit, pe_pos):
        self.position = pe_pos
        self.hw_config = hw_config

        rty, rtx = pe_pos[0], pe_pos[1]
        pey, pex = pe_pos[2], pe_pos[3]

        if rty % 2 == 0:
            self.plot_idx = rty * self.hw_config.Router_num_x * self.hw_config.PE_num + \
                            rtx * self.hw_config.PE_num + \
                            pey * self.hw_config.PE_num_x + \
                            pex
        else:
            self.plot_idx = rty * self.hw_config.Router_num_x * self.hw_config.PE_num + \
                            (self.hw_config.Router_num_x - rtx - 1) * self.hw_config.PE_num + \
                            pey * self.hw_config.PE_num_x + \
                            pex

        self.state = False

        size = self.hw_config.eDRAM_buffer_size * 1024 * 8 / input_bit
        self.edram_buffer = OnChipBuffer(size)
        self.buffer_size_util = [[], []] # [cycle, size]

        ### # Event queue's CU index
        self.edram_rd_cu_idx     = collections.deque() # set()
        self.cu_operation_cu_idx = collections.deque()

        ### event queue
        self.edram_rd_erp     = collections.deque()
        self.edram_rd_ir_erp  = [] # 一個CU一個edram_rd     event queue
        self.cu_operation_erp = [] # 一個CU一個cu_operation event queue
        for i in range(self.hw_config.CU_num):
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
        self.eDRAM_buffer_energy     = 0.
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

        ### CU state
        self.cu_state = []
        for i in range(self.hw_config.CU_num):
            self.cu_state.append(False)
        
        ### CU performance breakdown
        self.cu_performance_breakdown = []
        self.cu_busy_end_time = []
        for i in range(self.hw_config.CU_num):
            self.cu_performance_breakdown.append([0,0,0,0,0,0]) # [pre-layer, compute, transfer, overlap, other, total]
            self.cu_busy_end_time.append(1)

    def __str__(self):
        return str(self.__dict__)
