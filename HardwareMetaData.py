from configparser import ConfigParser

class HardwareMetaData(object):
    def __init__(self):
        cfg = ConfigParser()
        #cfg.read('./configs/default_hw.ini')
        cfg.read('./configs/hardware.ini')
        
        self.Router_num_y = int(cfg['general']['Router_num_y'])
        self.Router_num_x = int(cfg['general']['Router_num_x'])
        self.Router_num = self.Router_num_y * self.Router_num_x
        self.PE_num_y = int(cfg['general']['PE_num_y'])
        self.PE_num_x = int(cfg['general']['PE_num_x'])
        self.PE_num = self.PE_num_y * self.PE_num_x
        self.CU_num_y = int(cfg['general']['CU_num_y'])
        self.CU_num_x = int(cfg['general']['CU_num_x'])
        self.CU_num = self.CU_num_y * self.CU_num_x
        self.Xbar_num_y = int(cfg['general']['Xbar_num_y'])
        self.Xbar_num_x = int(cfg['general']['Xbar_num_x'])
        self.Xbar_num = self.Xbar_num_y * self.Xbar_num_x

        self.eDRAM_buffer_size = float(cfg['general']['eDRAM_buffer_size'])
        #self.eDRAM_buffer_bank_num = int(cfg['general']['eDRAM_buffer_bank_num'])
        #self.eDRAM_buffer_bus_width = int(cfg['general']['eDRAM_buffer_bus_width'])
        
        self.Output_Reg_size = int(cfg['general']['Output_Reg_size'])
        

        self.Xbar_h = int(cfg['general']['Xbar_h'])
        self.Xbar_w = int(cfg['general']['Xbar_w'])
        self.OU_h = int(cfg['general']['OU_h'])
        self.OU_w = int(cfg['general']['SA_num'])

        self.eDRAM_buffer_rd_per_cycle = self.Xbar_h * self.Xbar_num
        self.CU_Shift_and_add_per_cycle = int(cfg['general']['CU_Shift_and_add_per_cycle'])
        self.PE_Shift_and_add_per_cycle = int(cfg['general']['PE_Shift_and_add_per_cycle'])
        self.Activation_per_cycle = int(cfg['general']['Activation_per_cycle'])
        self.Pooling_per_cycle = int(cfg['general']['Pooling_per_cycle'])

        #self.bits_per_cell = int(cfg['general']['bits_per_cell'])
        #self.DAC_resolution = int(cfg['general']['DAC_resolution'])
        #self.SA_resolution = int(cfg['general']['SA_resolution'])

        self.CU_Input_Reg_size = int(cfg['general']['CU_Input_Reg_size'])
        self.CU_Output_Reg_size = int(cfg['general']['CU_Output_Reg_size'])

        # Leakage
        self.eDRAM_buffer_leakage = float(cfg['leakage']['eDRAM_buffer_leakage'])
        self.Router_leakage = float(cfg['leakage']['Router_leakage'])
        self.SA_leakage = float(cfg['leakage']['SA_leakage'])
        self.Act_leakage = float(cfg['leakage']['Act_leakage'])
        self.PE_SAA_leakage = float(cfg['leakage']['PE_SAA_leakage'])
        self.Pool_leakage = float(cfg['leakage']['Pool_leakage'])

        self.DAC_leakage = float(cfg['leakage']['DAC_leakage'])
        self.MUX_leakage = float(cfg['leakage']['MUX_leakage'])
        self.SA_leakage = float(cfg['leakage']['SA_leakage'])
        self.Crossbar_leakage = float(cfg['leakage']['Crossbar_leakage'])
        self.CU_SAA_leakage = float(cfg['leakage']['CU_SAA_leakage'])

        # Dynamic Energy
        self.edram_rd_ir_energy = float(cfg['dynamic_energy']['edram_rd_ir_energy'])
        self.edram_rd_pool_energy = float(cfg['dynamic_energy']['edram_rd_pool_energy'])
        self.ou_operation_energy = float(cfg['dynamic_energy']['ou_operation_energy'])
        self.pe_saa_energy = float(cfg['dynamic_energy']['pe_saa_energy'])
        self.cu_saa_energy = float(cfg['dynamic_energy']['cu_saa_energy'])
        self.activation_energy = float(cfg['dynamic_energy']['activation_energy'])
        self.pooling_energy = float(cfg['dynamic_energy']['pooling_energy'])
        self.edram_wr_energy = float(cfg['dynamic_energy']['edram_wr_energy'])
        self.router_energy = float(cfg['dynamic_energy']['router_energy'])

        self.pe_or_energy = float(cfg['dynamic_energy']['pe_or_energy'])
        self.cu_ir_energy = float(cfg['dynamic_energy']['cu_ir_energy'])
        self.cu_or_energy = float(cfg['dynamic_energy']['cu_or_energy'])
        
        self.dac_energy = float(cfg['dynamic_energy']['dac_energy'])
        self.xb_energy = float(cfg['dynamic_energy']['xb_energy'])
        self.mux_energy = float(cfg['dynamic_energy']['mux_energy'])
        self.sa_energy = float(cfg['dynamic_energy']['sa_energy'])
        self.sa_energy += self.mux_energy
       
    def __str__(self):
        return str(self.__dict__)




