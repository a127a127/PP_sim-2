from Model import Model
from MappingMetaData import MappingMetaData
import sys, csv
from math import ceil

class SCF(object):
    def __init__(self, model_config, hw_config, partition_h, partition_w, cant_use_pe):
        print("Model:", model_config.Model_type)
        self.model_info = Model(model_config)
        self.hw_config = hw_config
        self.RTY, self.RTX = cant_use_pe[0], cant_use_pe[1]
        self.PEY, self.PEX = cant_use_pe[2], cant_use_pe[3]

        self.cells_per_weight = ceil(self.model_info.filter_bit / self.hw_config.cell_bit_width)
        self.filters_per_xb = self.hw_config.Xbar_w // self.cells_per_weight
        self.filters_per_xb = ceil(self.filters_per_xb  / partition_w) 
        self.inputs_per_xb  = ceil(self.hw_config.Xbar_h / partition_h)

        self.mapping_to_xbar = [] # convolution and fully
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
        
        self.used_pe = set()
        self.last_pe = (0,0,0,0)
        self.next_pe = (0,0,0,0)
        
        self.layer_used_pe = [] # for Ordergenerator
        self.layer_used_xb_num = []
        
        self.map()

    def map(self):
        rt_h, rt_w = 0, 0
        pe_h, pe_w = 0, 0
        cu_n = 0
        xb_n = 0
        
        for nlayer in range(self.model_info.layer_length):
            self.layer_used_pe.append(set()) # for Ordergenerator
            self.layer_used_xb_num.append(0)
            if self.model_info.layer_list[nlayer].layer_type == "convolution":
                print("Convolution", nlayer)
                # for pooling
                pool_pe_id = (rt_h, rt_w, pe_h, pe_w)
                
                self.used_pe.add((rt_h, rt_w, pe_h, pe_w))
                self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))

                # Weight matrix
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
                        self.used_pe.add((rt_h, rt_w, pe_h, pe_w))
                        self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))
                        if h != mapping_height_num_xb-1:
                            Inp = [i for i in range(h * self.inputs_per_xb,(h+1) * self.inputs_per_xb)]
                        else:
                            Inp = [i for i in range(h * self.inputs_per_xb, matrix_height)]
                        self.mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_n][xb_n][nlayer].append(MappingMetaData(Inp, Cols, Filters))
                        
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

                self.used_pe.add((rt_h, rt_w, pe_h, pe_w))
                self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))


                # Weight matrix
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
                        self.used_pe.add((rt_h, rt_w, pe_h, pe_w))
                        self.last_pe = (rt_h, rt_w, pe_h, pe_w)
                        self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))
                        if h != mapping_height_num_xb-1:
                            Inp = [i for i in range(h * self.inputs_per_xb,(h+1) * self.inputs_per_xb)]
                        else:
                            Inp = [i for i in range(h * self.inputs_per_xb, matrix_height)]
                        self.mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_n][xb_n][nlayer].append(MappingMetaData(Inp, Cols, Filters))
                        
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
                self.next_pe = (rt_h, rt_w, pe_h, pe_w)
                 
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

class SRF(object):
    def __init__(self, model_config, hw_config, partition_h, partition_w, cant_use_pe):
        print("Model:", model_config.Model_type)
        self.model_info = Model(model_config)
        self.hw_config  = hw_config
        self.RTY, self.RTX = cant_use_pe[0], cant_use_pe[1]
        self.PEY, self.PEX = cant_use_pe[2], cant_use_pe[3]

        self.cells_per_weight = ceil(self.model_info.filter_bit / self.hw_config.cell_bit_width)
        self.filters_per_xb = self.hw_config.Xbar_w // self.cells_per_weight
        self.filters_per_xb = ceil(self.filters_per_xb  / partition_w) 
        self.inputs_per_xb  = ceil(self.hw_config.Xbar_h / partition_h)

        self.mapping_to_xbar = [] # convolution and fully
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
        
        self.used_pe = set()
        self.last_pe = (0,0,0,0)
        self.next_pe = (0,0,0,0)
        
        self.layer_used_pe = [] # for Ordergenerator
        self.layer_used_xb_num = []
        
        self.map()

    def map(self):
        rt_h, rt_w = 0, 0
        pe_h, pe_w = 0, 0
        cu_n = 0
        xb_n = 0
        
        for nlayer in range(self.model_info.layer_length):
            self.layer_used_pe.append(set()) # for Ordergenerator
            self.layer_used_xb_num.append(0)
            if self.model_info.layer_list[nlayer].layer_type == "convolution":
                print("Convolution", nlayer)
                # for pooling
                pool_pe_id = (rt_h, rt_w, pe_h, pe_w)

                self.used_pe.add((rt_h, rt_w, pe_h, pe_w))
                self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))
                
                # Weight matrix
                matrix_height = self.model_info.filter_length[nlayer]
                matrix_width  = self.model_info.filter_n[nlayer]
                
                mapping_width_num_xb  = ceil(matrix_width  / self.filters_per_xb)
                mapping_height_num_xb = ceil(matrix_height / self.inputs_per_xb)
                for h in range(mapping_height_num_xb):
                    if h != mapping_height_num_xb-1:
                        Inp = [i for i in range(h * self.inputs_per_xb,(h+1) * self.inputs_per_xb)]
                    else:
                        Inp = [i for i in range(h * self.inputs_per_xb, matrix_height)]
                    for w in range(mapping_width_num_xb):
                        self.used_pe.add((rt_h, rt_w, pe_h, pe_w))
                        self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))
                        if w + 1 == mapping_width_num_xb:
                            Filters = [i for i in range(w * self.filters_per_xb, matrix_width)]
                        else:
                            Filters = [i for i in range(w * self.filters_per_xb,(w+1) * self.filters_per_xb)]
                        Cols = len(Filters) * self.cells_per_weight
                        self.mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_n][xb_n][nlayer].append(MappingMetaData(Inp, Cols, Filters))
                        
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

                self.used_pe.add((rt_h, rt_w, pe_h, pe_w))
                self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))

                ## Weight matrix
                matrix_height = self.model_info.filter_length[nlayer]
                matrix_width  = self.model_info.filter_n[nlayer]

                mapping_width_num_xb  = ceil(matrix_width  / self.filters_per_xb)
                mapping_height_num_xb = ceil(matrix_height / self.inputs_per_xb)
                for h in range(mapping_height_num_xb):
                    if h != mapping_height_num_xb-1:
                        Inp = [i for i in range(h * self.inputs_per_xb,(h+1) * self.inputs_per_xb)]
                    else:
                        Inp = [i for i in range(h * self.inputs_per_xb, matrix_height)]
                    for w in range(mapping_width_num_xb):
                        self.used_pe.add((rt_h, rt_w, pe_h, pe_w))
                        self.last_pe = (rt_h, rt_w, pe_h, pe_w)
                        self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))
                        if w + 1 == mapping_width_num_xb:
                            Filters = [i for i in range(w * self.filters_per_xb, matrix_width)]
                        else:
                            Filters = [i for i in range(w * self.filters_per_xb,(w+1) * self.filters_per_xb)]
                        Cols = len(Filters) * self.cells_per_weight
                        self.mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_n][xb_n][nlayer].append(MappingMetaData(Inp, Cols, Filters))
                        
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
                self.next_pe = (rt_h, rt_w, pe_h, pe_w)

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
