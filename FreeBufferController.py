from configs.ModelConfig import ModelConfig
from Model import Model
from HardwareMetaData import HardwareMetaData
import numpy as np
import copy

class FreeBufferController(object):
    def __init__(self):
        model_info = Model(ModelConfig())
        hd_info = HardwareMetaData()
        pe_input_require = []
        pe_input_require.append(np.zeros((model_info.input_h[0]*model_info.input_w[0]*model_info.input_c[0])).tolist())
        for i in range(model_info.layer_length):
            h = model_info.input_h[i+1]
            w = model_info.input_w[i+1]
            c = model_info.input_c[i+1]
            pe_input_require.append(np.zeros((h * w * c)).tolist())

        self.input_require = []
        for rty in range(hd_info.Router_num_y):
            for rtx in range(hd_info.Router_num_x):
                for pey in range(hd_info.PE_num_y):
                    for pex in range(hd_info.PE_num_x):
                        self.input_require.append(copy.deepcopy(pe_input_require))

    def __str__(self):
        return str(self.__dict__)