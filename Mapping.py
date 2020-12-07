from Model import Model
from MappingMetaData import MappingMetaData
import sys, csv
from math import ceil
import numpy as np

class LIDR(object):
    def __init__(self, model_config, hw_config, partition_h, partition_w, cant_use_pe):
        print("Model:", model_config.Model_type)
        self.model_info = Model(model_config)
        self.hw_config = hw_config
        self.RTY, self.RTX = cant_use_pe[0], cant_use_pe[1]
        self.PEY, self.PEX = cant_use_pe[2], cant_use_pe[3]
        
        self.partition_size_h = ceil(self.hw_config.Xbar_h / partition_h)
        self.partition_size_w = ceil(self.hw_config.Xbar_w / partition_w)

        self.inputs_per_xb  = self.partition_size_h
        self.cells_per_weight = ceil(self.model_info.filter_bit / self.hw_config.cell_bit_width)
        self.filters_per_xb = self.partition_size_w  // self.cells_per_weight # 同filter放在同一個XB

        self.mapping_to_xbar = [] # convolution and fully # 紀錄XB對應的input data, 還有此input data與哪些column、filter做運算
        self.mapping_to_pe = [] # pooling
        for rty_idx in range(self.hw_config.Router_num_y):
            self.mapping_to_xbar.append([])
            self.mapping_to_pe.append([])
            for rtx_idx in range(self.hw_config.Router_num_x):
                self.mapping_to_xbar[rty_idx].append([])
                self.mapping_to_pe[rty_idx].append([])
                for pey_idx in range(self.hw_config.PE_num_y):
                    self.mapping_to_xbar[rty_idx][rtx_idx].append([])
                    self.mapping_to_pe[rty_idx][rtx_idx].append([])
                    for pex_idx in range(self.hw_config.PE_num_x):
                        self.mapping_to_xbar[rty_idx][rtx_idx][pey_idx].append([])
                        self.mapping_to_pe[rty_idx][rtx_idx][pey_idx].append([])
                        for nlayer in range(self.model_info.layer_length):
                            self.mapping_to_pe[rty_idx][rtx_idx][pey_idx][pex_idx].append([])
                        for cu_idx in range(self.hw_config.CU_num):
                            self.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx].append([])
                            for xb_idx in range(self.hw_config.Xbar_num):
                                self.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx].append([])
                                for nlayer in range(self.model_info.layer_length):
                                    self.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx].append([])
                                    if self.model_info.layer_list[nlayer].layer_type == "convolution":
                                        o_height = self.model_info.input_h[nlayer+1]
                                        o_width = self.model_info.input_w[nlayer+1]
                                        for window_h in range(o_height):
                                            self.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx][nlayer].append([])
                                            for window_w in range(o_width):
                                                self.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx][nlayer][window_h].append([])
        
        #self.used_pe = set()
        
        self.layer_used_pe = [] # for Ordergenerator
        
        self.map()

    def map(self):
        rt_h, rt_w = 0, 0
        pe_h, pe_w = 0, 0
        cu_n = 0
        xb_n = 0
        
        for nlayer in range(self.model_info.layer_length):
            self.layer_used_pe.append(set()) # for Ordergenerator
            if self.model_info.layer_list[nlayer].layer_type == "convolution":
                print("Convolution", nlayer)
                # for pooling
                pool_pe_id = (rt_h, rt_w, pe_h, pe_w)
                
                #self.used_pe.add((rt_h, rt_w, pe_h, pe_w))
                self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))
                
                ## Weight matrix
                matrix_height = self.model_info.filter_length[nlayer]
                matrix_width  = self.model_info.filter_n[nlayer]

                mapping_width_num_xb  = ceil(matrix_width  / self.filters_per_xb)
                mapping_height_num_xb = ceil(matrix_height / self.inputs_per_xb)
                
                for w in range(mapping_width_num_xb):
                    if w + 1 == mapping_width_num_xb:
                        Filters = [i for i in range(w * self.filters_per_xb, matrix_width)]
                    else:
                        Filters = [i for i in range(w * self.filters_per_xb,(w+1) * self.filters_per_xb)]
                    Cols = len(Filters) * self.cells_per_weight
                    for h in range(mapping_height_num_xb):
                        #self.used_pe.add((rt_h, rt_w, pe_h, pe_w))
                        self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))
                        if h != mapping_height_num_xb-1:
                            # pos = [i for i in range(h * self.inputs_per_xb,(h+1) * self.inputs_per_xb)]
                            start_pos = h * self.inputs_per_xb
                            end_pos = (h+1) * self.inputs_per_xb
                        else:
                            # pos = [i for i in range(h * self.inputs_per_xb, matrix_height)]
                            start_pos = h * self.inputs_per_xb
                            end_pos = matrix_height

                        ## Input vectors
                        # input_vetors = []
                        strides = self.model_info.strides[nlayer]
                        pad = self.model_info.pad[nlayer]
                        o_height = self.model_info.input_h[nlayer+1]
                        o_width = self.model_info.input_w[nlayer+1]
                        for window_h in range(o_height):
                            for window_w in range(o_width):
                                input_vector_data = []
                               #---此window的data---#
                                for c in range(self.model_info.filter_c[nlayer]):
                                    for h in range(self.model_info.filter_h[nlayer]):
                                        for w in range(self.model_info.filter_w[nlayer]):
                                            # padding後的位置
                                            pad_pos_h  = window_h*strides+h
                                            pad_pos_w  = window_w*strides+w
                                            # feature map的位置
                                            fm_h = pad_pos_h - pad
                                            fm_w = pad_pos_w - pad

                                            if fm_w >= 0 and fm_w < self.model_info.input_w[nlayer] and fm_h >= 0 and fm_h < self.model_info.input_h[nlayer]:
                                                input_vector_data.append((nlayer, fm_h, fm_w, c))
                                            else:
                                                input_vector_data.append(0) # padding的值為0
                                #input_vetors.append(input_vector_data)
                               #-------------------#
                                Inp = input_vector_data[start_pos:end_pos]
                                self.mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_n][xb_n][nlayer][window_h][window_w] = MappingMetaData(Inp, Cols, Filters)
                
                        # next crossbar position
                        xb_n += 1
                        if xb_n >= self.hw_config.Xbar_num:
                            xb_n = 0
                            cu_n += 1
                            if cu_n >= self.hw_config.CU_num:
                                cu_n = 0
                                pe_w += 1
                                if pe_w >= self.hw_config.PE_num_x:
                                    pe_w = 0
                                    pe_h += 1
                                    if pe_h >= self.hw_config.PE_num_y:
                                        pe_h = 0
                                        if rt_h % 2 == 0:
                                            rt_w += 1
                                            if rt_w >= self.hw_config.Router_num_x:
                                                rt_w -= 1
                                                rt_h += 1
                                                if rt_h >= self.hw_config.Router_num_y:
                                                    rt_h, rt_w = 0, 0
                                        else:
                                            rt_w -= 1
                                            if rt_w < 0:
                                                rt_w = 0
                                                rt_h += 1
                                                if rt_h >= self.hw_config.Router_num_y:
                                                    rt_h, rt_w = 0, 0

                                if rt_h == self.RTY and rt_w == self.RTX and pe_h == self.PEY and pe_w == self.PEX:
                                    rt_h, rt_w, pe_h, pe_w = 0, 0, 0, 0
                
                # next layer next PE
                if cu_n != 0 or xb_n != 0:
                    cu_n, xb_n = 0, 0
                    pe_w += 1
                    if pe_w >= self.hw_config.PE_num_x:
                        pe_w = 0
                        pe_h += 1
                        if pe_h >= self.hw_config.PE_num_y:
                            pe_h = 0
                            if rt_h % 2 == 0:
                                rt_w += 1
                                if rt_w >= self.hw_config.Router_num_x:
                                    rt_w -= 1
                                    rt_h += 1
                                    if rt_h >= self.hw_config.Router_num_y:
                                        rt_h, rt_w = 0, 0
                            else:
                                rt_w -= 1
                                if rt_w < 0:
                                    rt_w = 0
                                    rt_h += 1
                                    if rt_h >= self.hw_config.Router_num_y:
                                        rt_h, rt_w = 0, 0
                    if rt_h == self.RTY and rt_w == self.RTX and pe_h == self.PEY and pe_w == self.PEX:
                        rt_h, rt_w, pe_h, pe_w = 0, 0, 0, 0

            elif self.model_info.layer_list[nlayer].layer_type == "fully":
                print("Fully", nlayer)
                #self.used_pe.add((rt_h, rt_w, pe_h, pe_w))
                self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))

                ## Input vector
                input_vetor = []
                for h in range(self.model_info.filter_length[nlayer]):
                    input_vetor.append((nlayer, h, 0, 0))

                ## Weight matrix
                matrix_height = self.model_info.filter_length[nlayer]
                matrix_width  = self.model_info.filter_n[nlayer]

                mapping_width_num_xb  = ceil(matrix_width  / self.filters_per_xb)
                mapping_height_num_xb = ceil(matrix_height / self.inputs_per_xb)
                for w in range(mapping_width_num_xb):
                    if w + 1 == mapping_width_num_xb:
                        Filters = [i for i in range(w * self.filters_per_xb, matrix_width)]
                    else:
                        Filters = [i for i in range(w * self.filters_per_xb,(w+1) * self.filters_per_xb)]
                    Cols = len(Filters) * self.cells_per_weight
                    for h in range(mapping_height_num_xb):
                        #self.used_pe.add((rt_h, rt_w, pe_h, pe_w))
                        self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))
                        if h != mapping_height_num_xb-1:
                            # pos = [i for i in range(h * self.inputs_per_xb,(h+1) * self.inputs_per_xb)]
                            start_pos = h * self.inputs_per_xb
                            end_pos = (h+1) * self.inputs_per_xb
                        else:
                            # pos = [i for i in range(h * self.inputs_per_xb, matrix_height)]
                            start_pos = h * self.inputs_per_xb
                            end_pos = matrix_height

                        Inp = input_vetor[start_pos:end_pos]
                        self.mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_n][xb_n][nlayer] = MappingMetaData(Inp, Cols, Filters)
                        
                        # next crossbar position
                        xb_n += 1
                        if xb_n >= self.hw_config.Xbar_num:
                            xb_n = 0
                            cu_n += 1
                            if cu_n >= self.hw_config.CU_num:
                                cu_n = 0
                                pe_w += 1
                                if pe_w >= self.hw_config.PE_num_x:
                                    pe_w = 0
                                    pe_h += 1
                                    if pe_h >= self.hw_config.PE_num_y:
                                        pe_h = 0
                                        if rt_h % 2 == 0:
                                            rt_w += 1
                                            if rt_w >= self.hw_config.Router_num_x:
                                                rt_w -= 1
                                                rt_h += 1
                                                if rt_h >= self.hw_config.Router_num_y:
                                                    rt_h, rt_w = 0, 0
                                        else:
                                            rt_w -= 1
                                            if rt_w < 0:
                                                rt_w = 0
                                                rt_h += 1
                                                if rt_h >= self.hw_config.Router_num_y:
                                                    rt_h, rt_w = 0, 0
                                if rt_h == self.RTY and rt_w == self.RTX and pe_h == self.PEY and pe_w == self.PEX:
                                    rt_h, rt_w, pe_h, pe_w = 0, 0, 0, 0
                
                # next layer next PE
                if cu_n != 0 or xb_n != 0:
                    cu_n, xb_n = 0, 0
                    pe_w += 1
                    if pe_w >= self.hw_config.PE_num_x:
                        pe_w = 0
                        pe_h += 1
                        if pe_h >= self.hw_config.PE_num_y:
                            pe_h = 0
                            if rt_h % 2 == 0:
                                rt_w += 1
                                if rt_w >= self.hw_config.Router_num_x:
                                    rt_w -= 1
                                    rt_h += 1
                                    if rt_h >= self.hw_config.Router_num_y:
                                        rt_h, rt_w = 0, 0
                            else:
                                rt_w -= 1
                                if rt_w < 0:
                                    rt_w = 0
                                    rt_h += 1
                                    if rt_h >= self.hw_config.Router_num_y:
                                        rt_h, rt_w = 0, 0
                    if rt_h == self.RTY and rt_w == self.RTX and pe_h == self.PEY and pe_w == self.PEX:
                        rt_h, rt_w, pe_h, pe_w = 0, 0, 0, 0
                 
            elif self.model_info.layer_list[nlayer].layer_type == "pooling":
                print("Pooling", nlayer)
                next_layer_id = (rt_h, rt_w, pe_h, pe_w)
                rt_h, rt_w, pe_h, pe_w = pool_pe_id[0], pool_pe_id[1], pool_pe_id[2], pool_pe_id[3]
                self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))

                o_height  = self.model_info.input_h[nlayer+1]
                o_width   = self.model_info.input_w[nlayer+1]
                o_channel = self.model_info.input_c[nlayer+1]
                inputs = []
                for oh in range(o_height):
                    for ow in range(o_width):
                        for oc in range(o_channel):
                            nn = [(nlayer+1, oh, ow, oc), []]
                            for ph in range(self.model_info.pooling_h[nlayer]):
                                for pw in range(self.model_info.pooling_w[nlayer]):
                                    nn[1].append((nlayer, 
                                               oh * self.model_info.pooling_strides[nlayer] + ph,
                                               ow * self.model_info.pooling_strides[nlayer] + pw,
                                               oc))
                            inputs.append(nn)
                
                self.mapping_to_pe[rt_h][rt_w][pe_h][pe_w][nlayer] = inputs

                rt_h, rt_w, pe_h, pe_w = next_layer_id[0], next_layer_id[1], next_layer_id[2], next_layer_id[3]

    def __str__(self):
        return str(self.__dict__)

class HIDR(object):
    def __init__(self, model_config, hw_config, partition_h, partition_w, cant_use_pe):
        print("Model:", model_config.Model_type)
        self.model_info = Model(model_config)
        self.hw_config  = hw_config
        self.RTY, self.RTX = cant_use_pe[0], cant_use_pe[1]
        self.PEY, self.PEX = cant_use_pe[2], cant_use_pe[3]

        self.partition_size_h = ceil(self.hw_config.Xbar_h / partition_h)
        self.partition_size_w = ceil(self.hw_config.Xbar_w / partition_w)

        self.inputs_per_xb  = self.partition_size_h
        self.cells_per_weight = ceil(self.model_info.filter_bit / self.hw_config.cell_bit_width)
        self.filters_per_xb = self.partition_size_w  // self.cells_per_weight # 同filter放在同一個XB

        self.mapping_to_xbar = [] # convolution and fully # 紀錄XB對應的input data, 還有此input data與哪些column、filter做運算
        self.mapping_to_pe = [] # pooling 
        for rty_idx in range(self.hw_config.Router_num_y):
            self.mapping_to_xbar.append([])
            self.mapping_to_pe.append([])
            for rtx_idx in range(self.hw_config.Router_num_x):
                self.mapping_to_xbar[rty_idx].append([])
                self.mapping_to_pe[rty_idx].append([])
                for pey_idx in range(self.hw_config.PE_num_y):
                    self.mapping_to_xbar[rty_idx][rtx_idx].append([])
                    self.mapping_to_pe[rty_idx][rtx_idx].append([])
                    for pex_idx in range(self.hw_config.PE_num_x):
                        self.mapping_to_xbar[rty_idx][rtx_idx][pey_idx].append([])
                        self.mapping_to_pe[rty_idx][rtx_idx][pey_idx].append([])
                        for nlayer in range(self.model_info.layer_length):
                            self.mapping_to_pe[rty_idx][rtx_idx][pey_idx][pex_idx].append([])
                        for cu_idx in range(self.hw_config.CU_num):
                            self.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx].append([])
                            for xb_idx in range(self.hw_config.Xbar_num):
                                self.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx].append([])
                                for nlayer in range(self.model_info.layer_length):
                                    self.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx].append([])
                                    if self.model_info.layer_list[nlayer].layer_type == "convolution":
                                        o_height = self.model_info.input_h[nlayer+1]
                                        o_width = self.model_info.input_w[nlayer+1]
                                        for window_h in range(o_height):
                                            self.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx][nlayer].append([])
                                            for window_w in range(o_width):
                                                self.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx][nlayer][window_h].append([])
        
        #self.used_pe = set()
        
        self.layer_used_pe = [] # for Ordergenerator
        self.totoal_used_xb = set()
        self.ctr = 0
        
        self.map()

    def map(self):
        rt_h, rt_w = 0, 0
        pe_h, pe_w = 0, 0
        cu_n = 0
        xb_n = 0
        
        for nlayer in range(self.model_info.layer_length):
            self.layer_used_pe.append(set()) # for Ordergenerator
            if self.model_info.layer_list[nlayer].layer_type == "convolution":
                print("Convolution", nlayer)
                # for pooling
                pool_pe_id = (rt_h, rt_w, pe_h, pe_w)

                #self.used_pe.add((rt_h, rt_w, pe_h, pe_w))
                self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))
                
                ## Weight matrix
                matrix_height = self.model_info.filter_length[nlayer]
                matrix_width  = self.model_info.filter_n[nlayer]
                
                mapping_width_num_xb  = ceil(matrix_width  / self.filters_per_xb)
                mapping_height_num_xb = ceil(matrix_height / self.inputs_per_xb)
                
                

                for h in range(mapping_height_num_xb):
                    if h != mapping_height_num_xb-1:
                        # pos = [i for i in range(h * self.inputs_per_xb,(h+1) * self.inputs_per_xb)]
                        start_pos = h * self.inputs_per_xb
                        end_pos = (h+1) * self.inputs_per_xb
                    else:
                        # pos = [i for i in range(h * self.inputs_per_xb, matrix_height)]
                        start_pos = h * self.inputs_per_xb
                        end_pos = matrix_height
                        
                    for w in range(mapping_width_num_xb):
                        #self.used_pe.add((rt_h, rt_w, pe_h, pe_w))
                        self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))
                        if w + 1 == mapping_width_num_xb:
                            Filters = [i for i in range(w * self.filters_per_xb, matrix_width)]
                        else:
                            Filters = [i for i in range(w * self.filters_per_xb,(w+1) * self.filters_per_xb)]
                        Cols = len(Filters) * self.cells_per_weight
                        
                        ## Input vectors
                        # input_vetors = []
                        strides = self.model_info.strides[nlayer]
                        pad = self.model_info.pad[nlayer]
                        o_height = self.model_info.input_h[nlayer+1]
                        o_width = self.model_info.input_w[nlayer+1]
                        for window_h in range(o_height):
                            for window_w in range(o_width):
                                input_vector_data = []
                               #---此window的data---#
                                for c in range(self.model_info.filter_c[nlayer]):
                                    for h in range(self.model_info.filter_h[nlayer]):
                                        for w in range(self.model_info.filter_w[nlayer]):
                                            # padding後的位置
                                            pad_pos_h  = window_h*strides+h
                                            pad_pos_w  = window_w*strides+w
                                            # feature map的位置
                                            fm_h = pad_pos_h - pad
                                            fm_w = pad_pos_w - pad

                                            if fm_w >= 0 and fm_w < self.model_info.input_w[nlayer] and fm_h >= 0 and fm_h < self.model_info.input_h[nlayer]:
                                                input_vector_data.append((nlayer, fm_h, fm_w, c))
                                            else:
                                                input_vector_data.append(0) # padding的值為0
                               #-------------------#

                                Inp = input_vector_data[start_pos:end_pos]
                                self.mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_n][xb_n][nlayer][window_h][window_w] = MappingMetaData(Inp, Cols, Filters)

                        if (rt_h, rt_w, pe_h, pe_w, cu_n, xb_n) in self.totoal_used_xb:
                            self.ctr += 1
                        self.totoal_used_xb.add((rt_h, rt_w, pe_h, pe_w, cu_n, xb_n))
                        # next crossbar position
                        xb_n += 1
                        if xb_n >= self.hw_config.Xbar_num:
                                    xb_n = 0
                                    cu_n += 1
                                    if cu_n >= self.hw_config.CU_num:
                                        cu_n = 0
                                        pe_w += 1
                                        if pe_w >= self.hw_config.PE_num_x:
                                            pe_w = 0
                                            pe_h += 1
                                            if pe_h >= self.hw_config.PE_num_y:
                                                pe_h = 0
                                                if rt_h % 2 == 0:
                                                    rt_w += 1
                                                    if rt_w >= self.hw_config.Router_num_x:
                                                        rt_w -= 1
                                                        rt_h += 1
                                                        if rt_h >= self.hw_config.Router_num_y:
                                                            rt_h, rt_w = 0, 0
                                                else:
                                                    rt_w -= 1
                                                    if rt_w < 0:
                                                        rt_w = 0
                                                        rt_h += 1
                                                        if rt_h >= self.hw_config.Router_num_y:
                                                            rt_h, rt_w = 0, 0
                                        if rt_h == self.RTY and rt_w == self.RTX and pe_h == self.PEY and pe_w == self.PEX:
                                            rt_h, rt_w, pe_h, pe_w = 0, 0, 0, 0
                        
                # next layer next PE
                if cu_n != 0 or xb_n != 0:
                    cu_n, xb_n = 0, 0
                    pe_w += 1
                    if pe_w >= self.hw_config.PE_num_x:
                        pe_w = 0
                        pe_h += 1
                        if pe_h >= self.hw_config.PE_num_y:
                            pe_h = 0
                            if rt_h % 2 == 0:
                                rt_w += 1
                                if rt_w >= self.hw_config.Router_num_x:
                                    rt_w -= 1
                                    rt_h += 1
                                    if rt_h >= self.hw_config.Router_num_y:
                                        rt_h, rt_w = 0, 0
                            else:
                                rt_w -= 1
                                if rt_w < 0:
                                    rt_w = 0
                                    rt_h += 1
                                    if rt_h >= self.hw_config.Router_num_y:
                                        rt_h, rt_w = 0, 0
                    if rt_h == self.RTY and rt_w == self.RTX and pe_h == self.PEY and pe_w == self.PEX:
                        rt_h, rt_w, pe_h, pe_w = 0, 0, 0, 0

            elif self.model_info.layer_list[nlayer].layer_type == "fully":
                print("Fully", nlayer)

                #self.used_pe.add((rt_h, rt_w, pe_h, pe_w))
                self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))
                
                ## Input vector
                input_vetor = []
                for h in range(self.model_info.filter_length[nlayer]):
                    input_vetor.append((nlayer, h, 0, 0))

                ## Weight matrix
                matrix_height = self.model_info.filter_length[nlayer]
                matrix_width  = self.model_info.filter_n[nlayer]

                mapping_width_num_xb  = ceil(matrix_width  / self.filters_per_xb)
                mapping_height_num_xb = ceil(matrix_height / self.inputs_per_xb)
                for h in range(mapping_height_num_xb):
                    if h != mapping_height_num_xb-1:
                        # pos = [i for i in range(h * self.inputs_per_xb,(h+1) * self.inputs_per_xb)]
                        start_pos = h * self.inputs_per_xb
                        end_pos = (h+1) * self.inputs_per_xb
                    else:
                        # pos = [i for i in range(h * self.inputs_per_xb, matrix_height)]
                        start_pos = h * self.inputs_per_xb
                        end_pos = matrix_height

                    for w in range(mapping_width_num_xb):
                        #self.used_pe.add((rt_h, rt_w, pe_h, pe_w))
                        self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))
                        if w + 1 == mapping_width_num_xb:
                            Filters = [i for i in range(w * self.filters_per_xb, matrix_width)]
                        else:
                            Filters = [i for i in range(w * self.filters_per_xb,(w+1) * self.filters_per_xb)]
                        Cols = len(Filters) * self.cells_per_weight
                        
                        Inp = input_vetor[start_pos:end_pos]
                        self.mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_n][xb_n][nlayer] = MappingMetaData(Inp, Cols, Filters)
                        
                        if (rt_h, rt_w, pe_h, pe_w, cu_n, xb_n) in self.totoal_used_xb:
                            self.ctr += 1
                        self.totoal_used_xb.add((rt_h, rt_w, pe_h, pe_w, cu_n, xb_n))
                        
                        # next crossbar position
                        xb_n += 1
                        if xb_n >= self.hw_config.Xbar_num:
                            xb_n = 0
                            cu_n += 1
                            if cu_n >= self.hw_config.CU_num:
                                cu_n = 0
                                pe_w += 1
                                if pe_w >= self.hw_config.PE_num_x:
                                    pe_w = 0
                                    pe_h += 1
                                    if pe_h >= self.hw_config.PE_num_y:
                                        pe_h = 0
                                        if rt_h % 2 == 0:
                                            rt_w += 1
                                            if rt_w >= self.hw_config.Router_num_x:
                                                rt_w -= 1
                                                rt_h += 1
                                                if rt_h >= self.hw_config.Router_num_y:
                                                    rt_h, rt_w = 0, 0
                                        else:
                                            rt_w -= 1
                                            if rt_w < 0:
                                                rt_w = 0
                                                rt_h += 1
                                                if rt_h >= self.hw_config.Router_num_y:
                                                    rt_h, rt_w = 0, 0
                                
                                if rt_h == self.RTY and rt_w == self.RTX and pe_h == self.PEY and pe_w == self.PEX:
                                    rt_h, rt_w, pe_h, pe_w = 0, 0, 0, 0

                # next layer next PE
                if cu_n != 0 or xb_n != 0:
                    cu_n, xb_n = 0, 0
                    pe_w += 1
                    if pe_w >= self.hw_config.PE_num_x:
                        pe_w = 0
                        pe_h += 1
                        if pe_h >= self.hw_config.PE_num_y:
                            pe_h = 0
                            if rt_h % 2 == 0:
                                rt_w += 1
                                if rt_w >= self.hw_config.Router_num_x:
                                    rt_w -= 1
                                    rt_h += 1
                                    if rt_h >= self.hw_config.Router_num_y:
                                        rt_h, rt_w = 0, 0
                            else:
                                rt_w -= 1
                                if rt_w < 0:
                                    rt_w = 0
                                    rt_h += 1
                                    if rt_h >= self.hw_config.Router_num_y:
                                        rt_h, rt_w = 0, 0
                    if rt_h == self.RTY and rt_w == self.RTX and pe_h == self.PEY and pe_w == self.PEX:
                        rt_h, rt_w, pe_h, pe_w = 0, 0, 0, 0

            elif self.model_info.layer_list[nlayer].layer_type == "pooling":
                print("Pooling", nlayer)
                next_layer_id = (rt_h, rt_w, pe_h, pe_w)
                rt_h, rt_w, pe_h, pe_w = pool_pe_id[0], pool_pe_id[1], pool_pe_id[2], pool_pe_id[3]
                self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))

                o_height  = self.model_info.input_h[nlayer+1]
                o_width   = self.model_info.input_w[nlayer+1]
                o_channel = self.model_info.input_c[nlayer+1]
                inputs = []
                for oh in range(o_height):
                    for ow in range(o_width):
                        for oc in range(o_channel):
                            nn = [(nlayer+1, oh, ow, oc), []]
                            for ph in range(self.model_info.pooling_h[nlayer]):
                                for pw in range(self.model_info.pooling_w[nlayer]):
                                    nn[1].append((nlayer, 
                                               oh * self.model_info.pooling_strides[nlayer] + ph,
                                               ow * self.model_info.pooling_strides[nlayer] + pw,
                                               oc))
                            inputs.append(nn)
                
                self.mapping_to_pe[rt_h][rt_w][pe_h][pe_w][nlayer] = inputs
                rt_h, rt_w, pe_h, pe_w = next_layer_id[0], next_layer_id[1], next_layer_id[2], next_layer_id[3]

    def __str__(self):
        return str(self.__dict__)
