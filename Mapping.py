from Model import Model
from MappingMetaData import MappingMetaData
import sys, csv
from math import ceil
import numpy as np

class CaculateMappedCrossbarNum(object):
    def __init__(self, model_config, hw_config, partition_h, partition_w):
        self.model_info = Model(model_config)
        self.hw_config  = hw_config
        self.partition_h = partition_h
        self.partition_w = partition_w

        # mapping xb index
        rt_h, rt_w = 0, 0
        pe_h, pe_w = 0, 0
        cu_n, xb_n = 0, 0

        used_xb = 0

        for nlayer in range(self.model_info.layer_length):
            layer_type = self.model_info.layer_list[nlayer].layer_type
            if layer_type == "convolution" or layer_type == "fully":
                print(layer_type, nlayer)
                
                ## Weight matrix
                cells_per_weight = ceil(self.model_info.filter_bit / self.hw_config.cell_bit_width) # 16/2 = 8 cells per weight
                matrix_height    = self.model_info.filter_length[nlayer]
                matrix_width     = self.model_info.filter_n[nlayer]
                data_per_xb      = ceil(self.hw_config.Xbar_h / self.partition_h)
                filters_per_xb   = self.hw_config.Xbar_w // cells_per_weight
                filters_per_xb   = ceil(filters_per_xb / self.partition_w)
                
                # partitioning
                mapping_height_num_xb = ceil(matrix_height / data_per_xb)
                mapping_width_num_xb  = ceil(matrix_width  / filters_per_xb)
                for w in range(mapping_width_num_xb):
                    for h in range(mapping_height_num_xb):
                        # next crossbar position
                        xb_n += 1
                        used_xb += 1
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
     
            else:
                pass

        print("Router:", rt_h, rt_w)
        print("PE", pe_h, pe_w)
        print("CU", cu_n)
        print("XB", xb_n)
        print("Used XB:", used_xb)
        used_pe = ceil(used_xb/self.hw_config.Xbar_num/self.hw_config.CU_num)
        print("Used PE:", used_pe)
        used_rt = ceil(used_pe/4)
        print("Used Router:", used_rt)
        print("IDX", (rt_h, rt_w, pe_h, pe_w, cu_n, xb_n))

class LowInputReuseMapping(object):
    def __init__(self, model_config, hw_config, partition_h, partition_w, cant_use_xb):
        print("Model:", model_config.Model_type)
        self.model_info = Model(model_config)
        self.hw_config  = hw_config
        self.partition_h = partition_h
        self.partition_w = partition_w

        self.RTY, self.RTX = cant_use_xb[0], cant_use_xb[1]
        self.PEY, self.PEX = cant_use_xb[2], cant_use_xb[3]
        self.CUN, self.XBN = cant_use_xb[4], cant_use_xb[5]

        self.mapping_to_xbar = [] # convolution and fully crossbar operation
        self.mapping_to_pe = [] # pooling operation
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
        
        self.layer_used_pe = []
        self.layer_used_xb_num = []
        for nlayer in range(self.model_info.layer_length):
            self.layer_used_pe.append(set())
            self.layer_used_xb_num.append(0)
        
        self.map()

    def map(self):
        # mapping xb index
        rt_h, rt_w = 0, 0
        pe_h, pe_w = 0, 0
        cu_n, xb_n = 0, 0

        self.layer_mapped_xb = []
        
        for nlayer in range(self.model_info.layer_length):
            layer_type = self.model_info.layer_list[nlayer].layer_type
            if layer_type == "convolution" or layer_type == "fully":
                print(layer_type, nlayer)
                if layer_type == "convolution":
                    # for pooling
                    pool_pe_id = (rt_h, rt_w, pe_h, pe_w)

                self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))
                
                ## Weight matrix
                cells_per_weight = ceil(self.model_info.filter_bit / self.hw_config.cell_bit_width) # 16/2 = 8 cells per weight
                matrix_height    = self.model_info.filter_length[nlayer]
                matrix_width     = self.model_info.filter_n[nlayer]
                data_per_xb      = ceil(self.hw_config.Xbar_h / self.partition_h)
                filters_per_xb   = self.hw_config.Xbar_w // cells_per_weight
                filters_per_xb   = ceil(filters_per_xb / self.partition_w)
                
                # partitioning
                mapping_height_num_xb = ceil(matrix_height / data_per_xb)
                mapping_width_num_xb  = ceil(matrix_width  / filters_per_xb)
                for w in range(mapping_width_num_xb):
                    if w + 1 == mapping_width_num_xb:
                        Filters = [i for i in range(w * filters_per_xb, matrix_width)]
                    else:
                        Filters = [i for i in range(w * filters_per_xb,(w+1) * filters_per_xb)]
                    Cols = len(Filters) * cells_per_weight
                    for h in range(mapping_height_num_xb):
                        self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))
                        # map to a crossbar
                        if h != mapping_height_num_xb-1:
                            Inp = [i for i in range(h * data_per_xb,(h+1) * data_per_xb)]
                        else:
                            Inp = [i for i in range(h * data_per_xb, matrix_height)]
                        self.mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_n][xb_n][nlayer].append(MappingMetaData(Inp, Cols, Filters))
                        
                        # next crossbar position
                        xb_n += 1
                        self.layer_used_xb_num[nlayer] += 1
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
                        if (rt_h, rt_w, pe_h, pe_w, cu_n, xb_n) == (self.RTY, self.RTX, self.PEY, self.PEX, self.CUN, self.XBN):
                            rt_h, rt_w, pe_h, pe_w, cu_n, xb_n = 0, 0, 0, 0, 0, 0
     
            elif layer_type == "pooling":
                print("pooling", nlayer)
                next_layer_id = (rt_h, rt_w, pe_h, pe_w)
                rt_h, rt_w, pe_h, pe_w = pool_pe_id[0], pool_pe_id[1], pool_pe_id[2], pool_pe_id[3]

                o_height  = self.model_info.input_h[nlayer+1]
                o_width   = self.model_info.input_w[nlayer+1]
                o_channel = self.model_info.input_c[nlayer+1]
                inputs = []
                for oh in range(o_height):
                    for ow in range(o_width):
                        for oc in range(o_channel):
                            #num_input = oh * o_width + ow + c * o_height * o_width
                            nn = [(nlayer+1, oh, ow, oc), []]
                            for ph in range(self.model_info.pooling_h[nlayer]):
                                for pw in range(self.model_info.pooling_w[nlayer]):
                                    nn[1].append((nlayer, 
                                               oh * self.model_info.pooling_strides[nlayer] + ph,
                                               ow * self.model_info.pooling_strides[nlayer] + pw,
                                               oc))
                            inputs.append(nn)
                
                self.mapping_to_pe[rt_h][rt_w][pe_h][pe_w][nlayer] = inputs
                self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))

                rt_h, rt_w, pe_h, pe_w = next_layer_id[0], next_layer_id[1], next_layer_id[2], next_layer_id[3]

    def mapping_layout(self, path):
        with open(path+'/mapping_layout.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["layer", "PE_idx"])
            for nlayer in range(len(self.layer_used_pe)):
                for pe_idx in self.layer_used_pe[nlayer]:
                    rty, rtx, pey, pex = pe_idx[0], pe_idx[1], pe_idx[2], pe_idx[3]
                    if rty % 2 == 0:
                        plot_idx = rty * self.hw_config.Router_num_x * self.hw_config.PE_num + \
                                    rtx * self.hw_config.PE_num + \
                                    pey * self.hw_config.PE_num_x + \
                                    pex
                    else:
                        plot_idx = rty * self.hw_config.Router_num_x * self.hw_config.PE_num + \
                                    (self.hw_config.Router_num_x - rtx - 1) * self.hw_config.PE_num + \
                                    pey * self.hw_config.PE_num_x + \
                                    pex
                    writer.writerow([nlayer, plot_idx])

    def __str__(self):
        return str(self.__dict__)

class HighInputReuseMapping(object):
    def __init__(self, model_config, hw_config, partition_h, partition_w, cant_use_xb):
        print("Model:", model_config.Model_type)
        self.model_info = Model(model_config)
        self.hw_config = hw_config
        self.partition_h = partition_h
        self.partition_w = partition_w

        self.RTY, self.RTX = cant_use_xb[0], cant_use_xb[1]
        self.PEY, self.PEX = cant_use_xb[2], cant_use_xb[3]
        self.CUN, self.XBN = cant_use_xb[4], cant_use_xb[5]

        self.mapping_to_xbar = [] # convolution and fully crossbar operation
        self.mapping_to_pe = [] # pooling operation
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
        
        self.layer_used_pe = []
        self.layer_used_xb_num = []
        for nlayer in range(self.model_info.layer_length):
            self.layer_used_pe.append(set())
            self.layer_used_xb_num.append(0)
        
        self.map()

    def map(self):
        # mapping xb index
        rt_h, rt_w = 0, 0
        pe_h, pe_w = 0, 0
        cu_n, xb_n = 0, 0
        
        for nlayer in range(self.model_info.layer_length):
            layer_type = self.model_info.layer_list[nlayer].layer_type
            if layer_type == "convolution" or layer_type == "fully":
                print(layer_type, nlayer)
                if layer_type == "convolution":
                    # for pooling
                    pool_pe_id = (rt_h, rt_w, pe_h, pe_w)

                self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))
                
                ## Weight matrix
                cells_per_weight = ceil(self.model_info.filter_bit / self.hw_config.cell_bit_width) # 16/2 = 8 cells per weight
                matrix_height    = self.model_info.filter_length[nlayer]
                matrix_width     = self.model_info.filter_n[nlayer]
                data_per_xb      = ceil(self.hw_config.Xbar_h / self.partition_h)
                filters_per_xb   = self.hw_config.Xbar_w // cells_per_weight
                filters_per_xb   = ceil(filters_per_xb / self.partition_w)
                
                # partitioning
                mapping_height_num_xb = ceil(matrix_height / data_per_xb)
                mapping_width_num_xb  = ceil(matrix_width  / filters_per_xb)
                #for w in range(mapping_width_num_xb):
                for h in range(mapping_height_num_xb):
                    if h != mapping_height_num_xb-1:
                        Inp = [i for i in range(h * self.hw_config.Xbar_h,(h+1) * self.hw_config.Xbar_h)]
                    else:
                        Inp = [i for i in range(h * self.hw_config.Xbar_h, matrix_height)]
                    for w in range(mapping_width_num_xb):
                        self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))
                        # map to a crossbar
                        if w + 1 == mapping_width_num_xb:
                            Filters = [i for i in range(w * filters_per_xb, matrix_width)]
                        else:
                            Filters = [i for i in range(w * filters_per_xb,(w+1) * filters_per_xb)]
                        Cols = len(Filters) * cells_per_weight
                        self.mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_n][xb_n][nlayer].append(MappingMetaData(Inp, Cols, Filters))
                        
                        # next crossbar position
                        xb_n += 1
                        self.layer_used_xb_num[nlayer] += 1
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
                        if (rt_h, rt_w, pe_h, pe_w, cu_n, xb_n) == (self.RTY, self.RTX, self.PEY, self.PEX, self.CUN, self.XBN):
                            rt_h, rt_w, pe_h, pe_w, cu_n, xb_n = 0, 0, 0, 0, 0, 0
     
            elif layer_type == "pooling":
                print("pooling", nlayer)
                next_layer_id = (rt_h, rt_w, pe_h, pe_w)
                rt_h, rt_w, pe_h, pe_w = pool_pe_id[0], pool_pe_id[1], pool_pe_id[2], pool_pe_id[3]

                o_height  = self.model_info.input_h[nlayer+1]
                o_width   = self.model_info.input_w[nlayer+1]
                o_channel = self.model_info.input_c[nlayer+1]
                inputs = []
                for oh in range(o_height):
                    for ow in range(o_width):
                        for oc in range(o_channel):
                            #num_input = oh * o_width + ow + c * o_height * o_width
                            nn = [(nlayer+1, oh, ow, oc), []]
                            for ph in range(self.model_info.pooling_h[nlayer]):
                                for pw in range(self.model_info.pooling_w[nlayer]):
                                    nn[1].append((nlayer, 
                                               oh * self.model_info.pooling_strides[nlayer] + ph,
                                               ow * self.model_info.pooling_strides[nlayer] + pw,
                                               oc))
                            inputs.append(nn)
                
                self.mapping_to_pe[rt_h][rt_w][pe_h][pe_w][nlayer] = inputs
                self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))

                rt_h, rt_w, pe_h, pe_w = next_layer_id[0], next_layer_id[1], next_layer_id[2], next_layer_id[3]

    def mapping_layout(self, path):
        with open(path+'/mapping_layout.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["layer", "PE_idx"])
            for nlayer in range(len(self.layer_used_pe)):
                for pe_idx in self.layer_used_pe[nlayer]:
                    rty, rtx, pey, pex = pe_idx[0], pe_idx[1], pe_idx[2], pe_idx[3]
                    if rty % 2 == 0:
                        plot_idx = rty * self.hw_config.Router_num_x * self.hw_config.PE_num + \
                                    rtx * self.hw_config.PE_num + \
                                    pey * self.hw_config.PE_num_x + \
                                    pex
                    else:
                        plot_idx = rty * self.hw_config.Router_num_x * self.hw_config.PE_num + \
                                    (self.hw_config.Router_num_x - rtx - 1) * self.hw_config.PE_num + \
                                    pey * self.hw_config.PE_num_x + \
                                    pex
                    writer.writerow([nlayer, plot_idx])

    def __str__(self):
        return str(self.__dict__)
