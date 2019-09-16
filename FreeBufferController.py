from configs.ModelConfig import ModelConfig
from Model import Model
from HardwareMetaData import HardwareMetaData
import numpy as np

class FreeBufferController(object):
    def __init__(self):
        model_info = Model(ModelConfig())
        hd_info = HardwareMetaData()
        self.input_require = []
        for rty in range(hd_info.Router_num_y):
            for rtx in range(hd_info.Router_num_x):
                for pey in range(hd_info.PE_num_y):
                    for pex in range(hd_info.PE_num_x):
                        pe_input_require = []
                        pe_input_require.append(np.zeros((model_info.input_h[0]*model_info.input_w[0]*model_info.input_c[0])).tolist())
                        for i in range(model_info.layer_length):
                            if model_info.layer_list[i].layer_type == "convolution":
                                if i+1 < model_info.layer_length and model_info.layer_list[i+1].layer_type == "fully":
                                    pe_input_require.append(np.zeros((model_info.input_h[i+1] * model_info.input_w[i+1] * model_info.input_c[i+1])).tolist())
                                else:
                                    pe_input_require.append(np.zeros((model_info.input_h[i+1] * model_info.input_w[i+1] * model_info.input_c[i+1])).tolist())
                            elif model_info.layer_list[i].layer_type == "pooling":
                                if i+1 < model_info.layer_length and model_info.layer_list[i+1].layer_type == "fully":
                                    pe_input_require.append(np.zeros((model_info.input_h[i+1] * model_info.input_w[i+1] * model_info.input_c[i+1])).tolist())
                                else:
                                    pe_input_require.append(np.zeros((model_info.input_h[i+1] * model_info.input_w[i+1] * model_info.input_c[i+1])).tolist())
                            elif model_info.layer_list[i].layer_type == "fully":
                                pe_input_require.append(np.zeros((model_info.filter_n[i])).tolist())
                        self.input_require.append(pe_input_require)
    def __str__(self):
        return str(self.__dict__)