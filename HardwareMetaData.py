from configparser import ConfigParser

class HardwareMetaData(object):
    def __init__(self):
        # Ref: ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars

        cfg = ConfigParser()
        cfg.read('./configs/hardware.ini')

        self.cell_bit_width = 2  # Alexnet: 2

        self.Router_num_y = 9 # Alexnet: 9
        self.Router_num_x = 9 # Alexnet: 9
        self.Router_num = self.Router_num_y * self.Router_num_x
        self.PE_num_y = 2
        self.PE_num_x = 2
        self.PE_num = self.PE_num_y * self.PE_num_x
        self.CU_num_y = 4 # Alexnet: 4
        self.CU_num_x = 3 # Alexnet: 3
        self.CU_num = self.CU_num_y * self.CU_num_x
        self.Xbar_num_y = 4 # Alexnet: 4
        self.Xbar_num_x = 2 # Alexnet: 2
        self.Xbar_num = self.Xbar_num_y * self.Xbar_num_x
        self.Xbar_h = 128 #128 #10
        self.Xbar_w = 128 #128 #10
        self.OU_h = 9 #9 #5
        self.OU_w = 8 #8 #5

        self.Frequency = 1.2 # GHz
        self.ADC_resolution = 3 # bits
        self.Router_flit_size = 32 # bits
        
        self.cycle_time = 15.6 * (self.ADC_resolution/3) * (32/65) # scaling from 張老師 paper
        self.eDRAM_read_latency = 1 / 1.2 / 256 # scaling from ISAAC (ns/per bit)

        self.eDRAM_buffer_size = 1.6 # nKB
        self.Output_Reg_size = 3 # nKB
        self.CU_Input_Reg_size = 2 # nKB
        self.CU_Output_Reg_size = 256 # nKB
        
        self.CU_Shift_and_add_per_cycle = 4
        self.PE_Shift_and_add_per_cycle = 4
        self.Activation_per_cycle = 100
        self.Pooling_per_cycle = 100


        # Leakage
        self.eDRAM_buffer_leakage = 0
        self.Router_leakage = 0
        self.Act_leakage = 0
        self.PE_SAA_leakage = 0
        self.Pool_leakage = 0
        self.DAC_leakage = 0
        self.MUX_leakage = 0
        self.SA_leakage = 0
        self.Crossbar_leakage = 0
        self.CU_SAA_leakage = 0

        # Dynamic Energy (nJ)
        self.Energy_edram_buffer = 20.7 * 0.01 / 1.2 / 256 # per bit
        self.Energy_bus = 7 * 0.01 / 1.2 / 256 # per bit
        self.Energy_router = 42 * 0.01 / 1.2 / 32 / 4 # per bit
        self.Energy_activation = 0.26 * 0.01 / 1.2 # per data doing acivation function
        self.Energy_shift_and_add = 0.05 * 0.01 / 2 / 1.2 # per data doing shift and add
        self.Energy_pooling = 0.4 * 0.01 / 1.2 # nxn data doing pooling
        self.Energy_or = 1.68 * 0.01 / 1.2 / 128 # per bit
        self.Energy_adc = 16 * 0.01 / 1.2 / 8 * self.OU_w * \
                          (2**self.ADC_resolution / (self.ADC_resolution+1)) / (2**8/(8+1)) \
                          # per ou 
        self.Energy_dac = 4 * 0.01 / 1.2 / 8 * self.OU_h / 128 # per ou
        self.Energy_crossbar = 2.4 * 0.01 / 1.2 / 8 * ((self.OU_h * self.OU_w) / (128 * 128)) # per ou
        self.Energy_ir_in_cu = 1.24 * 0.01 / 1.2 / 256 # per bit
        self.Energy_or_in_cu = 0.23 * 0.01 / 1.2 / 128 # per bit

        # Off chip fetch
        self.Fetch_cycle = 1

    def __str__(self):
        return str(self.__dict__)
