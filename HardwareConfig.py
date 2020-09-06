
from math import floor

class HardwareConfig(object):
    def __init__(self):
        # Ref: ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars
        
        self.Router_num_y = 14
        self.Router_num_x = 13
        self.Router_num = self.Router_num_y * self.Router_num_x
        self.PE_num_y = 2
        self.PE_num_x = 2
        self.PE_num = self.PE_num_y * self.PE_num_x
        self.CU_num = 12
        self.Xbar_num = 8
        self.Xbar_h = 128
        self.Xbar_w = 128
        self.OU_h = 9
        self.OU_w = 8

        self.total_pe_num = self.Router_num * self.PE_num

        self.cell_bit_width = 2

        self.Frequency = 1.2 # GHz
        self.ADC_resolution = 3 # bits
        self.Router_flit_size = 32 # bits
        
        self.cycle_time = 15.6 * (self.ADC_resolution/3) * (32/65) # scaling from 張老師 paper # ns
        self.interconnect_step_num = int(self.cycle_time // self.Frequency)
        # assume data from a router to another router cost one cycle

        self.eDRAM_read_bits  = floor(256 * 1.2 * self.cycle_time) # bits / per cycle
        self.eDRAM_write_bits = floor(128 * 1.2 * self.cycle_time) # bits / per cycle

        self.eDRAM_buffer_size  = 64 # nKB

        # Leakage ()
        # self.eDRAM_buffer_leakage = 0
        # self.Router_leakage       = 0
        # self.Act_leakage          = 0
        # self.PE_SAA_leakage       = 0
        # self.Pool_leakage         = 0
        # self.DAC_leakage          = 0
        # self.MUX_leakage          = 0
        # self.SA_leakage           = 0
        # self.Crossbar_leakage     = 0
        # self.CU_SAA_leakage       = 0

        # Dynamic Energy (nJ)
        self.Energy_edram_buffer  = 20.7 / 1000 / 1.2 / 256 # per bit
        self.Energy_bus           = 7 / 1000 / 1.2 / 256 # per bit
        self.Energy_activation    = 0.26 / 1000 / 1.2 # per data doing acivation function
        self.Energy_shift_and_add = 0.05 / 1000 / 2 / 1.2 # per data doing shift and add
        self.Energy_pooling       = 0.4 / 1000 / 1.2 # nxn data doing pooling
        self.Energy_or            = 1.68 / 1000 / 1.2 / 128 # per bit
        # per operation unit (ou)
        self.Energy_ou_dac        = 4 / 1000 / 1.2 / 8 * self.OU_h / 128
        self.Energy_ou_crossbar   = 2.4 / 1000 / 1.2 / 8 * ((self.OU_h * self.OU_w) / (128 * 128))
        self.Energy_ou_adc        = 16 / 1000 / 1.2 / 8 * self.OU_w * \
                                  (2**self.ADC_resolution / (self.ADC_resolution+1)) / (2**8/(8+1))
        self.Energy_ou_ssa        = self.Energy_shift_and_add * self.OU_w
        self.Energy_ir_in_cu      = 1.24 / 1000 / 1.2 / 256 # per bit
        self.Energy_or_in_cu      = 0.23 / 1000 / 1.2 / 128 # per bit

        self.Energy_router        = 42 / 1000 / 1.2 / 32 / 4 # per bit # flit size = 32 # shared by 4 PEs
        self.Energy_link          = 10.4 / 1.6 / 32 / 77 # per bit # total links = 7*6+5*7=77


        # Off chip fetch: 目前還沒有資料
        self.Fetch_cycle  = 1 # per data
        self.Energy_fecth = 0 # per data
        
    def __str__(self):
        return str(self.__dict__)