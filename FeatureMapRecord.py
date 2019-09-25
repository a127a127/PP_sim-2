from configs.ModelConfig import ModelConfig
from Model import Model
from FeatureMapMetaData import FeatureMapMetaData
import numpy as np

class FeatureMapRecord(object):
    def __init__(self):
        self.model_info = Model(ModelConfig())
        self.feature_mat = []

        self.feature_mat.append(
            np.zeros((
                self.model_info.input_h[0], 
                self.model_info.input_w[0],
                self.model_info.input_c[0]
                )).tolist()
            )

        for i in range(self.model_info.layer_length):
            if self.model_info.layer_list[i].layer_type == "convolution":
                if i+1 < self.model_info.layer_length and self.model_info.layer_list[i+1].layer_type == "fully":
                    self.feature_mat.append(
                        np.zeros((
                            self.model_info.input_h[i+1] * self.model_info.input_w[i+1] * self.model_info.input_c[i+1], 
                            1, 
                            1
                            )).tolist()
                        )
                else:
                    self.feature_mat.append(
                        np.zeros((
                            self.model_info.input_h[i+1], 
                            self.model_info.input_w[i+1], 
                            self.model_info.input_c[i+1]
                            )).tolist()
                        )

            elif self.model_info.layer_list[i].layer_type == "pooling":
                if i+1 < self.model_info.layer_length and self.model_info.layer_list[i+1].layer_type == "fully":
                    self.feature_mat.append(
                        np.zeros((
                            self.model_info.input_h[i+1] * 
                            self.model_info.input_w[i+1] * 
                            self.model_info.input_c[i+1], 
                            1, 
                            1
                            )).tolist()
                        )
                else:
                    self.feature_mat.append(
                        np.zeros((
                            self.model_info.input_h[i+1], 
                            self.model_info.input_w[i+1], 
                            self.model_info.input_c[i+1]
                            )).tolist()
                        )

            elif self.model_info.layer_list[i].layer_type == "fully":
                self.feature_mat.append(np.zeros((self.model_info.filter_n[i], 1, 1)).tolist())

    def process(self, event, cycle, pos):
        fm_id = event.nlayer + 1
        input_w = self.model_info.input_w[event.nlayer]
        input_h = self.model_info.input_h[event.nlayer]
        filter_w = self.model_info.filter_w[event.nlayer]
        filter_h = self.model_info.filter_h[event.nlayer]

        if event.event_type == "ou":
            xb_pos = pos
            window_per_row = input_w - filter_w + 1 # stride = 1
            window_per_col = input_h - filter_h + 1 # stride = 1
            num_window = window_per_row * window_per_col
            if self.model_info.layer_list[event.nlayer].layer_type == "fully":
                for data in event.outputs: # [(num_input, input_bit), (filter_nfilter, filter_nbit)]
                    h = data[1][0]
                    w = 0
                    c = 0
                    if self.feature_mat[fm_id][h][w][c] == 0:
                        self.feature_mat[fm_id][h][w][c] = FeatureMapMetaData(cycle, xb_pos, -1)

            elif event.nlayer + 1 < self.model_info.layer_length and \
                self.model_info.layer_list[event.nlayer+1].layer_type == "fully":
                for data in event.outputs: # [(num_input, input_bit), (filter_nfilter, filter_nbit)]
                    num_input = data[0][0]
                    nfilter = data[1][0]
                    h = num_input + num_window * nfilter
                    w = 0
                    c = 0
                    if self.feature_mat[fm_id][h][w][c] == 0:
                        self.feature_mat[fm_id][h][w][c] = FeatureMapMetaData(cycle, xb_pos, -1)
            else:
                for data in event.outputs: # [(num_input, input_bit), (filter_nfilter, filter_nbit)]
                    num_input = data[0][0]
                    nfilter = data[1][0]
                    h = num_input // window_per_row
                    w = num_input % window_per_row
                    c = nfilter
                    if self.feature_mat[fm_id][h][w][c] == 0:
                        self.feature_mat[fm_id][h][w][c] = FeatureMapMetaData(cycle, xb_pos, -1)

        elif event.event_type == "pooling":
            pe_pos = pos
            if event.nlayer+1 < self.model_info.layer_length and \
                self.model_info.layer_list[event.nlayer+1].layer_type == "fully":
                h = event.outputs[0][1] + \
                    event.outputs[0][0]*self.model_info.input_w[event.nlayer+1] + \
                    event.outputs[0][2]*self.model_info.input_w[event.nlayer+1]*self.model_info.input_h[event.nlayer+1]
                w = 0
                c = 0
            else:
                h = event.outputs[0][0]
                w = event.outputs[0][1]
                c = event.outputs[0][2]
            if self.feature_mat[fm_id][h][w][c] == 0:
                self.feature_mat[fm_id][h][w][c] = FeatureMapMetaData(cycle, pe_pos, -1)

        elif event.event_type == "edram_wr":
            pe_pos = pos
            if len(event.outputs[0]) == 3: # 排除saa write
                h = event.inputs[0][0]
                w = event.inputs[0][1]
                c = event.inputs[0][2]
                self.feature_mat[fm_id][h][w][c].finish_compute = cycle
                self.feature_mat[fm_id][h][w][c].arrived_buffer[pe_pos] = cycle

        elif event.event_type == 'edram_rd_ir': # packet arrived
            pe_pos = pos
            fm_id = event.nlayer
            if event.nlayer == 0:
                for data in event.outputs:
                    h = data[1]
                    w = data[2]
                    c = data[3]
                    if self.feature_mat[fm_id][h][w][c] == 0:
                        self.feature_mat[fm_id][h][w][c] = FeatureMapMetaData(1, pe_pos, 1)
                    self.feature_mat[fm_id][h][w][c].arrived_buffer[pe_pos] = cycle
            else:
                if self.model_info.layer_list[event.nlayer].layer_type == "fully": # [num_input, h, w, c]
                    for data in event.outputs: 
                        h = data[1]
                        w = 0
                        c = 0
                        self.feature_mat[fm_id][h][w][c].arrived_buffer[pe_pos] = cycle
                else:
                    for data in event.outputs: # [num_input, h, w, c]
                        h = data[1]
                        w = data[2]
                        c = data[3]
                        self.feature_mat[fm_id][h][w][c].arrived_buffer[pe_pos] = cycle

    def __str__(self):
        return str(self.__dict__)
