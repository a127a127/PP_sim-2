
from math import floor

class HardwareConfig(object):
    def __init__(self):
        # Ref: ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars

        # Architecture
        self.Router_num_y = 14
        self.Router_num_x = 13
        self.Router_num = self.Router_num_y * self.Router_num_x
        self.PE_num_y = 2 # c-mesh
        self.PE_num_x = 2 # c-mesh
        self.PE_num = self.PE_num_y * self.PE_num_x
        self.CU_num = 12
        self.Xbar_num = 8
        self.Xbar_h = 128
        self.Xbar_w = 128
        self.OU_h = 9
        self.OU_w = 8    
        self.total_pe_num = self.Router_num * self.PE_num
        
        self.PE_frequency = 1.2 # GHz
        
        # on-chip eDRAM buffer
        self.eDRAM_buffer_size  = 64 # KB
        self.eDRAM_buffer_banks_num = 2
        self.eDRAM_buffer_bus_width = 256 # bits

        # bus
        self.bus_wires = 384

        # router
        self.router_frequency = 1.2 # GHz
        self.router_flit_size = 32 # bits
        self.router_ports = 8

        # activation
        self.activation_num = 2
        
        # shift-and-add
        self.shift_and_add_num_in_PE = 1

        # pooling
        self.pooling_num = 1

        # OR
        self.OR_size_in_PE = 3 # KB
        self.OR_bus_width = 128

        # CU
        self.DAC_num = 1024 # 8x128
        self.DAC_resolution = 1 
        self.crossbar_num = 8
        self.cell_bit_width = 2
        self.ADC_num = 8
        self.ADC_resolution = 3
        self.shift_and_add_num_in_CU = 4
        self.IR_size = 2 # KB
        self.OR_size_in_CU = 256 # B

        # links
        self.links_frequency = 1.6 # GHz
        self.links_bw = 6.4 # GB/s

        self.cycle_time = 7.68 # 15.6 * (self.ADC_resolution/3) * (32/65) # scaling from W. H. Chen's paper
        self.interconnect_step_num = int(self.cycle_time // self.router_frequency) # router frequency = PE frequency
        self.eDRAM_buffer_read_bits  = floor(256 * self.PE_frequency * self.cycle_time) # bits / per cycle
        # self.eDRAM_write_bits = floor(128 * 1.2 * self.cycle_time) # bits / per cycle

        # Power (W)
        self.Power_eDRAM_buffer = 20.7 / 1000
        self.Power_bus = 7 / 1000
        self.Power_activation = 0.52 / 1000
        self.Power_shift_and_add_in_PE = 0.05 / 1000
        self.Power_pooling = 0.4 / 1000
        self.Power_OR_in_PE = 1.68 / 1000
        self.Power_DAC = 4 / 1000
        self.Power_crossbar = 2.4 / 1000
        self.Power_ADC = 16 / 1000 * (2**self.ADC_resolution / (self.ADC_resolution+1)) / (2**8/(8+1))
        self.Power_IR = 1.24 / 1000
        self.Power_OR_in_CU = 0.23 / 1000
        self.Power_shift_and_add_in_CU = 0.4 / 1000
        self.Power_router = 42 / 1000
        self.Power_link = 10.4

        # Energy (nJ)
        self.Energy_edram_buffer  = self.Power_eDRAM_buffer / self.PE_frequency / self.eDRAM_buffer_bus_width
        self.Energy_bus           = self.Power_bus / self.PE_frequency / self.bus_wires
        self.Energy_activation    = self.Power_activation / self.PE_frequency / self.activation_num
        self.Energy_shift_and_add = self.Power_shift_and_add_in_PE / self.PE_frequency / self.shift_and_add_num_in_PE # per data
        # self.Energy_shift_and_add_in_PE = self.Power_shift_and_add_in_PE / self.PE_frequency / self.shift_and_add_num_in_PE
        # self.Energy_shift_and_add_in_CU = self.Power_shift_and_add_in_CU / self.PE_frequency / self.shift_and_add_num_in_CU
        self.Energy_pooling       = self.Power_pooling / self.PE_frequency / self.pooling_num
        self.Energy_or            = self.Power_OR_in_PE / self.PE_frequency / self.OR_bus_width
        # per operation unit (ou)
        self.Energy_ou_dac        = self.Power_DAC / self.PE_frequency / self.DAC_num * self.OU_h
        self.Energy_ou_crossbar   = self.Power_crossbar / self.PE_frequency / self.crossbar_num * ((self.OU_h * self.OU_w) / (self.Xbar_h * self.Xbar_w))
        self.Energy_ou_adc        = self.Power_ADC / self.PE_frequency / self.ADC_num * self.OU_w
        self.Energy_ou_ssa        = self.Energy_shift_and_add * self.OU_w
        self.Energy_ir_in_cu      = self.Power_IR / self.PE_frequency / self.eDRAM_buffer_bus_width
        self.Energy_or_in_cu      = self.Power_OR_in_CU / self.PE_frequency / 128

        self.Energy_router        = self.Power_router / self.PE_frequency / self.router_flit_size / 4 # flit size = 32 # shared by 4 PEs
        self.Energy_link          = self.Power_link / self.links_frequency / self.links_bw / 8 / 77 / 4 # 8 bits = 1 byte # total links = 7*6+5*7=77

        # Leakage
        # self.Leakage_eDRAM_buffer = 0
        # self.Leakage_bus = 0
        # self.Leakage_activation = 0
        # self.Leakage_shift_and_add = 0
        # self.Leakage_pooling = 0
        # self.Leakage_OR_in_PE = 0
        # self.Leakage_DAC = 0
        # self.Leakage_crossbar = 0
        # self.Leakage_ADC = 0
        # self.Leakage_IR =  0
        # self.Leakage_OR_in_CU = 0
        # self.Leakage_router = 0
        # self.Leakage_link = 0


        # Off chip fetch: no data
        self.Fetch_cycle  = 20
        
    def __str__(self):
        return str(self.__dict__)
