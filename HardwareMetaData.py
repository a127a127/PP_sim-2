from configparser import ConfigParser

class HardwareMetaData(object):
    def __init__(self):
        cfg = ConfigParser()
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
        self.Xbar_h = int(cfg['general']['Xbar_h'])
        self.Xbar_w = int(cfg['general']['Xbar_w'])
        self.OU_h = int(cfg['general']['OU_h'])
        self.OU_w = int(cfg['general']['SA_num'])

        self.eDRAM_buffer_size = int(cfg['general']['eDRAM_buffer_size'])
        self.eDRAM_buffer_bank_num = int(cfg['general']['eDRAM_buffer_bank_num'])
        self.eDRAM_buffer_bus_width = int(cfg['general']['eDRAM_buffer_bus_width'])
        
        self.Output_Reg_size = int(cfg['general']['Output_Reg_size'])


        self.eDRAM_buffer_rd_per_cycle = self.Xbar_h * self.Xbar_num
        self.CU_Shift_and_add_per_cycle = int(cfg['general']['CU_Shift_and_add_per_cycle'])
        self.PE_Shift_and_add_per_cycle = int(cfg['general']['PE_Shift_and_add_per_cycle'])
        self.Activation_per_cycle = int(cfg['general']['Activation_per_cycle'])
        self.Pooling_per_cycle = int(cfg['general']['Pooling_per_cycle'])

        self.bits_per_cell = int(cfg['general']['bits_per_cell'])
        self.DAC_resolution = int(cfg['general']['DAC_resolution'])
        self.SA_resolution = int(cfg['general']['SA_resolution'])

        self.CU_Input_Reg_size = int(cfg['general']['CU_Input_Reg_size'])
        self.CU_Output_Reg_size = int(cfg['general']['CU_Output_Reg_size'])

        # Power
        self.eDRAM_buffer_read_energy = float(cfg['power']['eDRAM_buffer_read_energy'])
        self.eDRAM_buffer_write_energy = float(cfg['power']['eDRAM_buffer_write_energy'])
        self.eDRAM_to_CU_bus_power  = float(cfg['power']['eDRAM_to_CU_bus_power'])
        self.Router_power = float(cfg['power']['Router_power'])
        self.Activation_power = float(cfg['power']['Activation_power'])
        self.Shift_and_add_power = float(cfg['power']['Shift_and_add_power'])
        self.Pooling_power = float(cfg['power']['Pooling_power'])
        self.Output_Reg_power = float(cfg['power']['Output_Reg_power'])
        self.DAC_power = float(cfg['power']['DAC_power'])
        self.SA_power = float(cfg['power']['SA_power'])
        self.OU_power = float(cfg['power']['OU_power'])

    def __str__(self):
        return str(self.__dict__)




