from Model import Model
from IdleMetaData import IdleMetaData
import numpy as np

class IdleAnalysis(object):
    def __init__(self, model_config):
        self.model_info = Model(model_config)
        self.feature_mat = [] # layer n input_feature map
        #self.feature_mat.append(np.zeros((self.model_info.input_h[0], self.model_info.input_w[0], self.model_info.input_c[0])).tolist())

        for i in range(self.model_info.layer_length):
            if self.model_info.layer_list[i].layer_type == "convolution":
                if i+1 < self.model_info.layer_length and self.model_info.layer_list[i+1].layer_type == "fully":
                    self.feature_mat.append(np.zeros((self.model_info.input_h[i+1] * self.model_info.input_w[i+1] * self.model_info.input_c[i+1], 1, 1)).tolist())
                else:
                    self.feature_mat.append(np.zeros((self.model_info.input_h[i+1], self.model_info.input_w[i+1], self.model_info.input_c[i+1])).tolist())
            elif self.model_info.layer_list[i].layer_type == "pooling":
                if i+1 < self.model_info.layer_length and self.model_info.layer_list[i+1].layer_type == "fully":
                    self.feature_mat.append(np.zeros((self.model_info.input_h[i+1] * self.model_info.input_w[i+1] * self.model_info.input_c[i+1], 1, 1)).tolist())
                else:
                    self.feature_mat.append(np.zeros((self.model_info.input_h[i+1], self.model_info.input_w[i+1], self.model_info.input_c[i+1])).tolist())
            elif self.model_info.layer_list[i].layer_type == "fully":
                    self.feature_mat.append(np.zeros((self.model_info.filter_n[i], 1, 1)).tolist())

    def process(self, event, cycle):
        #print(event.event_type)
        nlayer = event.nlayer
        input_w = self.model_info.input_w[event.nlayer]
        input_h = self.model_info.input_h[event.nlayer]
        filter_w = self.model_info.filter_w[event.nlayer]
        filter_h = self.model_info.filter_h[event.nlayer]
        window_per_row = input_w - filter_w + 1 # slide 1, stride
        window_per_col = input_h - filter_h + 1 # slide 1, stride
        num_window = window_per_row * window_per_col

        if event.event_type == "ou":
            for data in event.outputs:
                # [(num_input, input_bit), (filter_nfilter, filter_nbit)]
                num_input = data[0][0]
                nfilter = data[1][0]
                # print("num_input:", num_input, "nfilter:", nfilter)
                
                if nlayer+1 < self.model_info.layer_length:
                    if self.model_info.layer_list[nlayer+1].layer_type == "convolution":
                        h = num_input // window_per_row
                        w = num_input % window_per_row
                        c = nfilter
                    elif self.model_info.layer_list[nlayer+1].layer_type == "fully":
                        h = num_input + num_window * nfilter
                        w = 0
                        c = 0
                    else:
                        print("layer type error:", )
                        exit(0)
                else:
                    if self.model_info.layer_list[nlayer].layer_type == "convolution":
                        h = num_input // window_per_row
                        w = num_input % window_per_row
                        c = nfilter
                    elif self.model_info.layer_list[nlayer].layer_type == "fully":
                        h = num_input
                        w = 0
                        c = 0
                    else:
                        print("layer type error:", )
                        exit(0)

                if self.feature_mat[nlayer][h][w][c] == 0: 
                    # (start_compute, finish_compute, finish_transfer)
                    self.feature_mat[nlayer][h][w][c] = IdleMetaData(cycle, -1)
        
        elif event.event_type == "edram_wr":
            if not event.outputs:
                # saa edram write
                return
            else:
                if nlayer+1 < self.model_info.layer_length:
                    h = event.outputs[0][0]
                    w = event.outputs[0][1]
                    c = event.outputs[0][2]
                    
                    self.feature_mat[nlayer][h][w][c].finish_compute = cycle

    def __str__(self):
        return str(self.__dict__)