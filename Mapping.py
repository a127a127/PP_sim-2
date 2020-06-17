from HardwareMetaData import HardwareMetaData as HW
from ModelConfig import ModelConfig
from Model import Model
from MappingMetaData import MappingMetaData
import sys
from math import ceil
import numpy as np

class SameColumnFirstMapping(object):
    def __init__(self):
        # 同一筆data要放在同一個crossbar
        model_config = ModelConfig()
        print("Model:", model_config.Model_type)
        self.model_info = Model(model_config)

        self.mapping_to_xbar = [] # convolution and fully
        self.mapping_to_pe = [] # pooling
        for rty_idx in range(HW().Router_num_y):
            self.mapping_to_xbar.append([])
            self.mapping_to_pe.append([])
            for rtx_idx in range(HW().Router_num_x):
                self.mapping_to_xbar[rty_idx].append([])
                self.mapping_to_pe[rty_idx].append([])
                for pey_idx in range(HW().PE_num_y):
                    self.mapping_to_xbar[rty_idx][rtx_idx].append([])
                    self.mapping_to_pe[rty_idx][rtx_idx].append([])
                    for pex_idx in range(HW().PE_num_x):
                        self.mapping_to_xbar[rty_idx][rtx_idx][pey_idx].append([])
                        self.mapping_to_pe[rty_idx][rtx_idx][pey_idx].append([])
                        for nlayer in range(self.model_info.layer_length):
                            self.mapping_to_pe[rty_idx][rtx_idx][pey_idx][pex_idx].append([])
                        for cu_idx in range(HW().CU_num):
                            self.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx].append([])
                            for xb_idx in range(HW().Xbar_num):
                                self.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx].append([])
                                for nlayer in range(self.model_info.layer_length):
                                    self.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx].append([])
        self.layer_used_pe = []
        for nlayer in range(self.model_info.layer_length):
            self.layer_used_pe.append([])
        self.map()

    def map(self):
        rt_h, rt_w = 0, 0
        pe_h, pe_w = 0, 0
        cu_n = 0
        xb_n = 0
        
        for nlayer in range(self.model_info.layer_length):
            if self.model_info.layer_list[nlayer].layer_type == "convolution":
                print("Convolution", nlayer)
                # for pooling
                pool_pe_id = (rt_h, rt_w, pe_h, pe_w)
                used_pe_num = 1

                self.layer_used_pe[nlayer].append((rt_h, rt_w, pe_h, pe_w))

                ## Weight matrix
                cells_per_weight = ceil(self.model_info.filter_bit / HW().cell_bit_width) # 16/2 = 8 cells per weight
                matrix_height = self.model_info.filter_length[nlayer]
                matrix_width  = self.model_info.filter_n[nlayer]
                filters_per_xb = HW().Xbar_w // cells_per_weight

                mapping_height_num_xb = ceil(matrix_height / HW().Xbar_h)
                mapping_width_num_xb  = ceil(matrix_width  / filters_per_xb) # width / 16
                for w in range(mapping_width_num_xb):
                    # Filters
                    if w + 1 == mapping_width_num_xb: # 邊界
                        Filters = [i for i in range(w * filters_per_xb, matrix_width)]
                    else:
                        Filters = [i for i in range(w * filters_per_xb,(w+1) * filters_per_xb)]
                    Cols = len(Filters) * cells_per_weight
                    
                    for h in range(mapping_height_num_xb):
                        # 一次map一個xb
                        if h != mapping_height_num_xb-1:
                            Inp = [i for i in range(h * HW().Xbar_h,(h+1) * HW().Xbar_h)]
                        else:
                            Inp = [i for i in range(h * HW().Xbar_h, matrix_height)]
                        self.mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_n][xb_n][nlayer].append(MappingMetaData(Inp, Cols, Filters))
                        
                        # 算下一個XB的位置
                        xb_n += 1
                        if xb_n >= HW().Xbar_num:
                            xb_n = 0
                            cu_n += 1
                            if cu_n >= HW().CU_num:
                                cu_n = 0
                                pe_w += 1
                                used_pe_num += 1
                                if pe_w >= HW().PE_num_x:
                                    pe_w = 0
                                    pe_h += 1
                                    if pe_h >= HW().PE_num_y:
                                        pe_h = 0
                                        if rt_h % 2 == 0:
                                            rt_w += 1
                                            if rt_w >= HW().Router_num_x:
                                                rt_w -= 1
                                                rt_h += 1
                                                if rt_h >= HW().Router_num_y:
                                                    print("no enough crossbar")
                                                    exit()
                                        else:
                                            rt_w -= 1
                                            if rt_w < 0:
                                                rt_w = 0
                                                rt_h += 1
                                                if rt_h >= HW().Router_num_y:
                                                    print("no enough crossbar")
                                                    exit()
                                self.layer_used_pe[nlayer].append((rt_h, rt_w, pe_h, pe_w))
                
                # next layer next PE
                if cu_n != 0 or xb_n != 0:
                    pe_w += 1
                    if pe_w >= HW().PE_num_x:
                        pe_w = 0
                        pe_h += 1
                        if pe_h >= HW().PE_num_y:
                            pe_h = 0
                            if rt_h % 2 == 0:
                                rt_w += 1
                                if rt_w >= HW().Router_num_x:
                                    rt_w -= 1
                                    rt_h += 1
                                    if rt_h >= HW().Router_num_y:
                                        print("no enough crossbar")
                                        exit()
                            else:
                                rt_w -= 1
                                if rt_w < 0:
                                    rt_w = 0
                                    rt_h += 1
                                    if rt_h >= HW().Router_num_y:
                                        print("no enough crossbar")
                                        exit()
                else:
                    used_pe_num -= 1
                    self.layer_used_pe[nlayer].remove((rt_h, rt_w, pe_h, pe_w))
        
            elif self.model_info.layer_list[nlayer].layer_type == "fully":
                print("Fully", nlayer)
                used_pe_num = 1
                self.layer_used_pe[nlayer].append((rt_h, rt_w, pe_h, pe_w))

                ## Weight matrix
                cells_per_weight = ceil(self.model_info.filter_bit / HW().cell_bit_width) # 16/2 = 8 cells per weight
                matrix_height = self.model_info.filter_length[nlayer]
                matrix_width  = self.model_info.filter_n[nlayer]
                filters_per_xb = HW().Xbar_w // cells_per_weight

                mapping_height_num_xb = ceil(matrix_height / HW().Xbar_h)
                mapping_width_num_xb  = ceil(matrix_width  / filters_per_xb) # width / 16
                for w in range(mapping_width_num_xb):
                    # Filters
                    if w + 1 == mapping_width_num_xb:
                        # 邊界
                        Filters = [i for i in range(w * filters_per_xb, matrix_width)]
                    else:
                        Filters = [i for i in range(w * filters_per_xb,(w+1) * filters_per_xb)]
                    Cols = len(Filters) * cells_per_weight

                    for h in range(mapping_height_num_xb):
                        # 一次map一個xb
                        if h != mapping_height_num_xb-1:
                            Inp = [i for i in range(h * HW().Xbar_h,(h+1) * HW().Xbar_h)]
                        else:
                            Inp = [i for i in range(h * HW().Xbar_h, matrix_height)]
                        self.mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_n][xb_n][nlayer].append(MappingMetaData(Inp, Cols, Filters))
                        
                        # 算下一個XB的位置
                        xb_n += 1
                        if xb_n >= HW().Xbar_num:
                            xb_n = 0
                            cu_n += 1
                            if cu_n >= HW().CU_num:
                                cu_n = 0
                                pe_w += 1
                                used_pe_num += 1
                                if pe_w >= HW().PE_num_x:
                                    pe_w = 0
                                    pe_h += 1
                                    if pe_h >= HW().PE_num_y:
                                        pe_h = 0
                                        if rt_h % 2 == 0:
                                            rt_w += 1
                                            if rt_w >= HW().Router_num_x:
                                                rt_w -= 1
                                                rt_h += 1
                                                if rt_h >= HW().Router_num_y:
                                                    print("no enough crossbar")
                                                    exit()
                                        else:
                                            rt_w -= 1
                                            if rt_w < 0:
                                                rt_w = 0
                                                rt_h += 1
                                                if rt_h >= HW().Router_num_y:
                                                    print("no enough crossbar")
                                                    exit()
                                self.layer_used_pe[nlayer].append((rt_h, rt_w, pe_h, pe_w))                
                
                # next layer next PE
                if cu_n != 0 or xb_n != 0:
                    pe_w += 1
                    if pe_w >= HW().PE_num_x:
                        pe_w = 0
                        pe_h += 1
                        if pe_h >= HW().PE_num_y:
                            pe_h = 0
                            if rt_h % 2 == 0:
                                rt_w += 1
                                if rt_w >= HW().Router_num_x:
                                    rt_w -= 1
                                    rt_h += 1
                                    if rt_h >= HW().Router_num_y:
                                        print("no enough crossbar")
                                        exit()
                            else:
                                rt_w -= 1
                                if rt_w < 0:
                                    rt_w = 0
                                    rt_h += 1
                                    if rt_h >= HW().Router_num_y:
                                        print("no enough crossbar")
                                        exit()
                else:
                    used_pe_num -= 1
                    self.layer_used_pe[nlayer].remove((rt_h, rt_w, pe_h, pe_w))
                    
            elif self.model_info.layer_list[nlayer].layer_type == "pooling":
                print("Pooling", nlayer)
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
                self.layer_used_pe[nlayer].append((rt_h, rt_w, pe_h, pe_w))

                rt_h, rt_w, pe_h, pe_w = next_layer_id[0], next_layer_id[1], next_layer_id[2], next_layer_id[3]
                
    def __str__(self):
        return str(self.__dict__)

class SameRowFirstMapping(object):
    def __init__(self):
        # 同一筆data要放在同一個crossbar
        model_config = ModelConfig()
        print("Model:", model_config.Model_type)
        self.model_info = Model(model_config)

        self.mapping_to_xbar = [] # convolution and fully
        self.mapping_to_pe = [] # pooling
        for rty_idx in range(HW().Router_num_y):
            self.mapping_to_xbar.append([])
            self.mapping_to_pe.append([])
            for rtx_idx in range(HW().Router_num_x):
                self.mapping_to_xbar[rty_idx].append([])
                self.mapping_to_pe[rty_idx].append([])
                for pey_idx in range(HW().PE_num_y):
                    self.mapping_to_xbar[rty_idx][rtx_idx].append([])
                    self.mapping_to_pe[rty_idx][rtx_idx].append([])
                    for pex_idx in range(HW().PE_num_x):
                        self.mapping_to_xbar[rty_idx][rtx_idx][pey_idx].append([])
                        self.mapping_to_pe[rty_idx][rtx_idx][pey_idx].append([])
                        for nlayer in range(self.model_info.layer_length):
                            self.mapping_to_pe[rty_idx][rtx_idx][pey_idx][pex_idx].append([])
                        for cu_idx in range(HW().CU_num):
                            self.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx].append([])
                            for xb_idx in range(HW().Xbar_num):
                                self.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx].append([])
                                for nlayer in range(self.model_info.layer_length):
                                    self.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx].append([])
        self.layer_used_pe = []
        for nlayer in range(self.model_info.layer_length):
            self.layer_used_pe.append([])
        self.map()

    def map(self):
        rt_h, rt_w = 0, 0
        pe_h, pe_w = 0, 0
        cu_n = 0
        xb_n = 0
        
        for nlayer in range(self.model_info.layer_length):
            if self.model_info.layer_list[nlayer].layer_type == "convolution":
                print("Convolution", nlayer)
                # for pooling
                pool_pe_id = (rt_h, rt_w, pe_h, pe_w)
                used_pe_num = 1

                self.layer_used_pe[nlayer].append((rt_h, rt_w, pe_h, pe_w))

                ## Weight matrix
                cells_per_weight = ceil(self.model_info.filter_bit / HW().cell_bit_width) # 16/2 = 8 cells per weight
                matrix_height = self.model_info.filter_length[nlayer]
                matrix_width  = self.model_info.filter_n[nlayer]
                filters_per_xb = HW().Xbar_w // cells_per_weight

                mapping_height_num_xb = ceil(matrix_height / HW().Xbar_h)
                mapping_width_num_xb  = ceil(matrix_width  / filters_per_xb) # width / 16
                
                for h in range(mapping_height_num_xb):
                    if h != mapping_height_num_xb-1:
                        Inp = [i for i in range(h * HW().Xbar_h,(h+1) * HW().Xbar_h)]
                    else:
                        Inp = [i for i in range(h * HW().Xbar_h, matrix_height)]
                    for w in range(mapping_width_num_xb):
                        # 一次map一個xb
                        if w + 1 == mapping_width_num_xb:
                            Filters = [i for i in range(w * filters_per_xb, matrix_width)]
                        else:
                            Filters = [i for i in range(w * filters_per_xb,(w+1) * filters_per_xb)]
                        Cols = len(Filters) * cells_per_weight
                        self.mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_n][xb_n][nlayer].append(MappingMetaData(Inp, Cols, Filters))
                        
                        # 算下一個XB的位置
                        xb_n += 1
                        if xb_n >= HW().Xbar_num:
                            xb_n = 0
                            cu_n += 1
                            if cu_n >= HW().CU_num:
                                cu_n = 0
                                pe_w += 1
                                used_pe_num += 1
                                if pe_w >= HW().PE_num_x:
                                    pe_w = 0
                                    pe_h += 1
                                    if pe_h >= HW().PE_num_y:
                                        pe_h = 0
                                        if rt_h % 2 == 0:
                                            rt_w += 1
                                            if rt_w >= HW().Router_num_x:
                                                rt_w -= 1
                                                rt_h += 1
                                                if rt_h >= HW().Router_num_y:
                                                    print("no enough crossbar")
                                                    exit()
                                        else:
                                            rt_w -= 1
                                            if rt_w < 0:
                                                rt_w = 0
                                                rt_h += 1
                                                if rt_h >= HW().Router_num_y:
                                                    print("no enough crossbar")
                                                    exit()
                                self.layer_used_pe[nlayer].append((rt_h, rt_w, pe_h, pe_w))
                
                # next layer next PE
                if cu_n != 0 or xb_n != 0:
                    pe_w += 1
                    if pe_w >= HW().PE_num_x:
                        pe_w = 0
                        pe_h += 1
                        if pe_h >= HW().PE_num_y:
                            pe_h = 0
                            if rt_h % 2 == 0:
                                rt_w += 1
                                if rt_w >= HW().Router_num_x:
                                    rt_w -= 1
                                    rt_h += 1
                                    if rt_h >= HW().Router_num_y:
                                        print("no enough crossbar")
                                        exit()
                            else:
                                rt_w -= 1
                                if rt_w < 0:
                                    rt_w = 0
                                    rt_h += 1
                                    if rt_h >= HW().Router_num_y:
                                        print("no enough crossbar")
                                        exit()
                else:
                    used_pe_num -= 1
                    self.layer_used_pe[nlayer].remove((rt_h, rt_w, pe_h, pe_w))
        
            elif self.model_info.layer_list[nlayer].layer_type == "fully":
                print("Fully", nlayer)
                sed_pe_num = 1
                self.layer_used_pe[nlayer].append((rt_h, rt_w, pe_h, pe_w))

                ## Weight matrix
                cells_per_weight = ceil(self.model_info.filter_bit / HW().cell_bit_width) # 16/2 = 8 cells per weight
                matrix_height = self.model_info.filter_length[nlayer]
                matrix_width  = self.model_info.filter_n[nlayer]
                filters_per_xb = HW().Xbar_w // cells_per_weight

                mapping_height_num_xb = ceil(matrix_height / HW().Xbar_h)
                mapping_width_num_xb  = ceil(matrix_width  / filters_per_xb) # width / 16
                for h in range(mapping_height_num_xb):
                    if h != mapping_height_num_xb-1:
                        Inp = [i for i in range(h * HW().Xbar_h,(h+1) * HW().Xbar_h)]
                    else:
                        Inp = [i for i in range(h * HW().Xbar_h, matrix_height)]
                    for w in range(mapping_width_num_xb):
                        # 一次map一個xb
                        if w + 1 == mapping_width_num_xb:
                            Filters = [i for i in range(w * filters_per_xb, matrix_width)]
                        else:
                            Filters = [i for i in range(w * filters_per_xb,(w+1) * filters_per_xb)]
                        Cols = len(Filters) * cells_per_weight
                        self.mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_n][xb_n][nlayer].append(MappingMetaData(Inp, Cols, Filters))
                        
                        # 算下一個XB的位置
                        xb_n += 1
                        if xb_n >= HW().Xbar_num:
                            xb_n = 0
                            cu_n += 1
                            if cu_n >= HW().CU_num:
                                cu_n = 0
                                pe_w += 1
                                used_pe_num += 1
                                if pe_w >= HW().PE_num_x:
                                    pe_w = 0
                                    pe_h += 1
                                    if pe_h >= HW().PE_num_y:
                                        pe_h = 0
                                        if rt_h % 2 == 0:
                                            rt_w += 1
                                            if rt_w >= HW().Router_num_x:
                                                rt_w -= 1
                                                rt_h += 1
                                                if rt_h >= HW().Router_num_y:
                                                    print("no enough crossbar")
                                                    exit()
                                        else:
                                            rt_w -= 1
                                            if rt_w < 0:
                                                rt_w = 0
                                                rt_h += 1
                                                if rt_h >= HW().Router_num_y:
                                                    print("no enough crossbar")
                                                    exit()
                                self.layer_used_pe[nlayer].append((rt_h, rt_w, pe_h, pe_w))
                
                # next layer next PE
                if cu_n != 0 or xb_n != 0:
                    pe_w += 1
                    if pe_w >= HW().PE_num_x:
                        pe_w = 0
                        pe_h += 1
                        if pe_h >= HW().PE_num_y:
                            pe_h = 0
                            if rt_h % 2 == 0:
                                rt_w += 1
                                if rt_w >= HW().Router_num_x:
                                    rt_w -= 1
                                    rt_h += 1
                                    if rt_h >= HW().Router_num_y:
                                        print("no enough crossbar")
                                        exit()
                            else:
                                rt_w -= 1
                                if rt_w < 0:
                                    rt_w = 0
                                    rt_h += 1
                                    if rt_h >= HW().Router_num_y:
                                        print("no enough crossbar")
                                        exit()
                else:
                    used_pe_num -= 1
                    self.layer_used_pe[nlayer].remove((rt_h, rt_w, pe_h, pe_w))
                
            elif self.model_info.layer_list[nlayer].layer_type == "pooling":
                print("Pooling", nlayer)
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
                self.layer_used_pe[nlayer].append((rt_h, rt_w, pe_h, pe_w))

                rt_h, rt_w, pe_h, pe_w = next_layer_id[0], next_layer_id[1], next_layer_id[2], next_layer_id[3]
                
    def __str__(self):
        return str(self.__dict__)

class SCFParallelsimMapping(object):
    def __init__(self, parall):
        self.Parall = parall
        model_config = ModelConfig()
        print("Model:", model_config.Model_type)
        self.model_info = Model(model_config)

        self.mapping_to_xbar = [] # convolution and fully
        self.mapping_to_pe = [] # pooling
        for rty_idx in range(HW().Router_num_y):
            self.mapping_to_xbar.append([])
            self.mapping_to_pe.append([])
            for rtx_idx in range(HW().Router_num_x):
                self.mapping_to_xbar[rty_idx].append([])
                self.mapping_to_pe[rty_idx].append([])
                for pey_idx in range(HW().PE_num_y):
                    self.mapping_to_xbar[rty_idx][rtx_idx].append([])
                    self.mapping_to_pe[rty_idx][rtx_idx].append([])
                    for pex_idx in range(HW().PE_num_x):
                        self.mapping_to_xbar[rty_idx][rtx_idx][pey_idx].append([])
                        self.mapping_to_pe[rty_idx][rtx_idx][pey_idx].append([])
                        for nlayer in range(self.model_info.layer_length):
                            self.mapping_to_pe[rty_idx][rtx_idx][pey_idx][pex_idx].append([])
                        for cu_idx in range(HW().CU_num):
                            self.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx].append([])
                            for xb_idx in range(HW().Xbar_num):
                                self.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx].append([])
                                for nlayer in range(self.model_info.layer_length):
                                    self.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx].append([])
        self.layer_used_pe = []
        for nlayer in range(self.model_info.layer_length):
            self.layer_used_pe.append([])
        self.map()

    def map(self):
        rt_h, rt_w = 0, 0
        pe_h, pe_w = 0, 0
        cu_n = 0
        xb_n = 0
        
        for nlayer in range(self.model_info.layer_length):
            if self.model_info.layer_list[nlayer].layer_type == "convolution":
                print("Convolution", nlayer)
                # for pooling
                pool_pe_id = (rt_h, rt_w, pe_h, pe_w)
                used_pe_num = 1

                self.layer_used_pe[nlayer].append((rt_h, rt_w, pe_h, pe_w))
                
                ## Weight matrix
                cells_per_weight = ceil(self.model_info.filter_bit / HW().cell_bit_width) # 16/2 = 8 cells per weight
                matrix_height = self.model_info.filter_length[nlayer]
                matrix_width  = self.model_info.filter_n[nlayer]
                filters_per_xb = HW().Xbar_w // cells_per_weight
                filters_per_xb = ceil(filters_per_xb / self.Parall) # 1

                H = ceil(HW().Xbar_h / self.Parall) # 2
                mapping_height_num_xb = ceil(matrix_height / H) # 3
                mapping_width_num_xb  = ceil(matrix_width  / filters_per_xb) # width / 16
                for w in range(mapping_width_num_xb):
                    if w + 1 == mapping_width_num_xb:
                        Filters = [i for i in range(w * filters_per_xb, matrix_width)]
                    else:
                        Filters = [i for i in range(w * filters_per_xb,(w+1) * filters_per_xb)]
                    Cols = len(Filters) * cells_per_weight
                    for h in range(mapping_height_num_xb):
                        # 一次map一個xb
                        if h != mapping_height_num_xb-1:
                            Inp = [i for i in range(h * H,(h+1) * H)]
                        else:
                            Inp = [i for i in range(h * H, matrix_height)]
                        self.mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_n][xb_n][nlayer].append(MappingMetaData(Inp, Cols, Filters))
                        
                        # 算下一個XB的位置
                        xb_n += 1
                        if xb_n >= HW().Xbar_num:
                            xb_n = 0
                            cu_n += 1
                            if cu_n >= HW().CU_num:
                                cu_n = 0
                                pe_w += 1
                                used_pe_num += 1
                                if pe_w >= HW().PE_num_x:
                                    pe_w = 0
                                    pe_h += 1
                                    if pe_h >= HW().PE_num_y:
                                        pe_h = 0
                                        if rt_h % 2 == 0:
                                            rt_w += 1
                                            if rt_w >= HW().Router_num_x:
                                                rt_w -= 1
                                                rt_h += 1
                                                if rt_h >= HW().Router_num_y:
                                                    rt_h, rt_w = 0, 0
                                                    #print("no enough crossbar")
                                                    #exit()
                                        else:
                                            rt_w -= 1
                                            if rt_w < 0:
                                                rt_w = 0
                                                rt_h += 1
                                                if rt_h >= HW().Router_num_y:
                                                    rt_h, rt_w = 0, 0
                                                    #print("no enough crossbar")
                                                    #exit()
                                self.layer_used_pe[nlayer].append((rt_h, rt_w, pe_h, pe_w))
                
                # next layer next PE
                if cu_n != 0 or xb_n != 0:
                    pe_w += 1
                    if pe_w >= HW().PE_num_x:
                        pe_w = 0
                        pe_h += 1
                        if pe_h >= HW().PE_num_y:
                            pe_h = 0
                            if rt_h % 2 == 0:
                                rt_w += 1
                                if rt_w >= HW().Router_num_x:
                                    rt_w -= 1
                                    rt_h += 1
                                    if rt_h >= HW().Router_num_y:
                                        rt_h, rt_w = 0, 0
                                        # print("no enough crossbar")
                                        # exit()
                            else:
                                rt_w -= 1
                                if rt_w < 0:
                                    rt_w = 0
                                    rt_h += 1
                                    if rt_h >= HW().Router_num_y:
                                        rt_h, rt_w = 0, 0
                                        # print("no enough crossbar")
                                        # exit()
                else:
                    used_pe_num -= 1
                    self.layer_used_pe[nlayer].remove((rt_h, rt_w, pe_h, pe_w))

            elif self.model_info.layer_list[nlayer].layer_type == "fully":
                print("Fully", nlayer)
                used_pe_num = 1
                self.layer_used_pe[nlayer].append((rt_h, rt_w, pe_h, pe_w))

                ## Weight matrix
                cells_per_weight = ceil(self.model_info.filter_bit / HW().cell_bit_width) # 16/2 = 8 cells per weight
                matrix_height = self.model_info.filter_length[nlayer]
                matrix_width  = self.model_info.filter_n[nlayer]
                filters_per_xb = HW().Xbar_w // cells_per_weight
                filters_per_xb = ceil(filters_per_xb / self.Parall) # 1

                H = ceil(HW().Xbar_h / self.Parall) # 2
                mapping_height_num_xb = ceil(matrix_height / H) # 3
                mapping_width_num_xb  = ceil(matrix_width  / filters_per_xb) # width / 16
                for w in range(mapping_width_num_xb):
                    if w + 1 == mapping_width_num_xb:
                        Filters = [i for i in range(w * filters_per_xb, matrix_width)]
                    else:
                        Filters = [i for i in range(w * filters_per_xb,(w+1) * filters_per_xb)]
                    Cols = len(Filters) * cells_per_weight
                    for h in range(mapping_height_num_xb):
                        # 一次map一個xb
                        if h != mapping_height_num_xb-1:
                            Inp = [i for i in range(h * H,(h+1) * H)]
                        else:
                            Inp = [i for i in range(h * H, matrix_height)]
                        self.mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_n][xb_n][nlayer].append(MappingMetaData(Inp, Cols, Filters))
                        
                        # 算下一個XB的位置
                        xb_n += 1
                        if xb_n >= HW().Xbar_num:
                            xb_n = 0
                            cu_n += 1
                            if cu_n >= HW().CU_num:
                                cu_n = 0
                                pe_w += 1
                                used_pe_num += 1
                                if pe_w >= HW().PE_num_x:
                                    pe_w = 0
                                    pe_h += 1
                                    if pe_h >= HW().PE_num_y:
                                        pe_h = 0
                                        if rt_h % 2 == 0:
                                            rt_w += 1
                                            if rt_w >= HW().Router_num_x:
                                                rt_w -= 1
                                                rt_h += 1
                                                if rt_h >= HW().Router_num_y:
                                                    rt_h, rt_w = 0, 0
                                                    # print("no enough crossbar")
                                                    # exit()
                                        else:
                                            rt_w -= 1
                                            if rt_w < 0:
                                                rt_w = 0
                                                rt_h += 1
                                                if rt_h >= HW().Router_num_y:
                                                    rt_h, rt_w = 0, 0
                                                    # print("no enough crossbar")
                                                    # exit()
                                self.layer_used_pe[nlayer].append((rt_h, rt_w, pe_h, pe_w))                
                
                # next layer next PE
                if cu_n != 0 or xb_n != 0:
                    pe_w += 1
                    if pe_w >= HW().PE_num_x:
                        pe_w = 0
                        pe_h += 1
                        if pe_h >= HW().PE_num_y:
                            pe_h = 0
                            if rt_h % 2 == 0:
                                rt_w += 1
                                if rt_w >= HW().Router_num_x:
                                    rt_w -= 1
                                    rt_h += 1
                                    if rt_h >= HW().Router_num_y:
                                        rt_h, rt_w = 0, 0
                                        # print("no enough crossbar")
                                        # exit()
                            else:
                                rt_w -= 1
                                if rt_w < 0:
                                    rt_w = 0
                                    rt_h += 1
                                    if rt_h >= HW().Router_num_y:
                                        rt_h, rt_w = 0, 0
                                        # print("no enough crossbar")
                                        # exit()
                else:
                    used_pe_num -= 1
                    self.layer_used_pe[nlayer].remove((rt_h, rt_w, pe_h, pe_w))
                 
            elif self.model_info.layer_list[nlayer].layer_type == "pooling":
                print("Pooling", nlayer)
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
                self.layer_used_pe[nlayer].append((rt_h, rt_w, pe_h, pe_w))

                rt_h, rt_w, pe_h, pe_w = next_layer_id[0], next_layer_id[1], next_layer_id[2], next_layer_id[3]

    def __str__(self):
        return str(self.__dict__)

class SRFParallelsimMapping(object):
    def __init__(self, parall):
        self.Parall = parall
        model_config = ModelConfig()
        print("Model:", model_config.Model_type)
        self.model_info = Model(model_config)

        self.mapping_to_xbar = [] # convolution and fully
        self.mapping_to_pe = [] # pooling
        for rty_idx in range(HW().Router_num_y):
            self.mapping_to_xbar.append([])
            self.mapping_to_pe.append([])
            for rtx_idx in range(HW().Router_num_x):
                self.mapping_to_xbar[rty_idx].append([])
                self.mapping_to_pe[rty_idx].append([])
                for pey_idx in range(HW().PE_num_y):
                    self.mapping_to_xbar[rty_idx][rtx_idx].append([])
                    self.mapping_to_pe[rty_idx][rtx_idx].append([])
                    for pex_idx in range(HW().PE_num_x):
                        self.mapping_to_xbar[rty_idx][rtx_idx][pey_idx].append([])
                        self.mapping_to_pe[rty_idx][rtx_idx][pey_idx].append([])
                        for nlayer in range(self.model_info.layer_length):
                            self.mapping_to_pe[rty_idx][rtx_idx][pey_idx][pex_idx].append([])
                        for cu_idx in range(HW().CU_num):
                            self.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx].append([])
                            for xb_idx in range(HW().Xbar_num):
                                self.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx].append([])
                                for nlayer in range(self.model_info.layer_length):
                                    self.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx].append([])
        self.layer_used_pe = []
        for nlayer in range(self.model_info.layer_length):
            self.layer_used_pe.append([])
        self.map()

    def map(self):
        rt_h, rt_w = 0, 0
        pe_h, pe_w = 0, 0
        cu_n = 0
        xb_n = 0
        
        for nlayer in range(self.model_info.layer_length):
            if self.model_info.layer_list[nlayer].layer_type == "convolution":
                print("Convolution", nlayer)
                # for pooling
                pool_pe_id = (rt_h, rt_w, pe_h, pe_w)
                used_pe_num = 1

                self.layer_used_pe[nlayer].append((rt_h, rt_w, pe_h, pe_w))

                ## Weight matrix
                cells_per_weight = ceil(self.model_info.filter_bit / HW().cell_bit_width) # 16/2 = 8 cells per weight
                matrix_height = self.model_info.filter_length[nlayer]
                matrix_width  = self.model_info.filter_n[nlayer]
                filters_per_xb = HW().Xbar_w // cells_per_weight
                filters_per_xb = ceil(filters_per_xb / self.Parall) # 1

                H = ceil(HW().Xbar_h / self.Parall) # 2
                mapping_height_num_xb = ceil(matrix_height / H) # 3
                mapping_width_num_xb  = ceil(matrix_width  / filters_per_xb) # width / 16
                for h in range(mapping_height_num_xb):
                    if h != mapping_height_num_xb-1:
                        Inp = [i for i in range(h * H,(h+1) * H)]
                    else:
                        Inp = [i for i in range(h * H, matrix_height)]
                    for w in range(mapping_width_num_xb):
                        # 一次map一個xb
                        if w + 1 == mapping_width_num_xb:
                            Filters = [i for i in range(w * filters_per_xb, matrix_width)]
                        else:
                            Filters = [i for i in range(w * filters_per_xb,(w+1) * filters_per_xb)]
                        Cols = len(Filters) * cells_per_weight
                        self.mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_n][xb_n][nlayer].append(MappingMetaData(Inp, Cols, Filters))
                        
                        # 算下一個XB的位置
                        xb_n += 1
                        if xb_n >= HW().Xbar_num:
                            xb_n = 0
                            cu_n += 1
                            if cu_n >= HW().CU_num:
                                cu_n = 0
                                pe_w += 1
                                used_pe_num += 1
                                if pe_w >= HW().PE_num_x:
                                    pe_w = 0
                                    pe_h += 1
                                    if pe_h >= HW().PE_num_y:
                                        pe_h = 0
                                        if rt_h % 2 == 0:
                                            rt_w += 1
                                            if rt_w >= HW().Router_num_x:
                                                rt_w -= 1
                                                rt_h += 1
                                                if rt_h >= HW().Router_num_y:
                                                    rt_h, rt_w = 0, 0
                                                    # print("no enough crossbar")
                                                    # exit()
                                        else:
                                            rt_w -= 1
                                            if rt_w < 0:
                                                rt_w = 0
                                                rt_h += 1
                                                if rt_h >= HW().Router_num_y:
                                                    rt_h, rt_w = 0, 0
                                                    # print("no enough crossbar")
                                                    # exit()
                                self.layer_used_pe[nlayer].append((rt_h, rt_w, pe_h, pe_w))
                
                # next layer next PE
                if cu_n != 0 or xb_n != 0:
                    pe_w += 1
                    if pe_w >= HW().PE_num_x:
                        pe_w = 0
                        pe_h += 1
                        if pe_h >= HW().PE_num_y:
                            pe_h = 0
                            if rt_h % 2 == 0:
                                rt_w += 1
                                if rt_w >= HW().Router_num_x:
                                    rt_w -= 1
                                    rt_h += 1
                                    if rt_h >= HW().Router_num_y:
                                        rt_h, rt_w = 0, 0
                                        # print("no enough crossbar")
                                        # exit()
                            else:
                                rt_w -= 1
                                if rt_w < 0:
                                    rt_w = 0
                                    rt_h += 1
                                    if rt_h >= HW().Router_num_y:
                                        rt_h, rt_w = 0, 0
                                        # print("no enough crossbar")
                                        # exit()
                else:
                    used_pe_num -= 1
                    self.layer_used_pe[nlayer].remove((rt_h, rt_w, pe_h, pe_w))

            elif self.model_info.layer_list[nlayer].layer_type == "fully":
                print("Fully", nlayer)
                used_pe_num = 1
                self.layer_used_pe[nlayer].append((rt_h, rt_w, pe_h, pe_w))

                ## Weight matrix
                cells_per_weight = ceil(self.model_info.filter_bit / HW().cell_bit_width) # 16/2 = 8 cells per weight
                matrix_height = self.model_info.filter_length[nlayer]
                matrix_width  = self.model_info.filter_n[nlayer]
                filters_per_xb = HW().Xbar_w // cells_per_weight
                filters_per_xb = ceil(filters_per_xb / self.Parall) # 1

                H = ceil(HW().Xbar_h / self.Parall) # 2
                mapping_height_num_xb = ceil(matrix_height / H) # 3
                mapping_width_num_xb  = ceil(matrix_width  / filters_per_xb) # width / 16
                for h in range(mapping_height_num_xb):
                    if h != mapping_height_num_xb-1:
                        Inp = [i for i in range(h * H,(h+1) * H)]
                    else:
                        Inp = [i for i in range(h * H, matrix_height)]
                    for w in range(mapping_width_num_xb):
                        # 一次map一個xb
                        if w + 1 == mapping_width_num_xb:
                            Filters = [i for i in range(w * filters_per_xb, matrix_width)]
                        else:
                            Filters = [i for i in range(w * filters_per_xb,(w+1) * filters_per_xb)]
                        Cols = len(Filters) * cells_per_weight
                        self.mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_n][xb_n][nlayer].append(MappingMetaData(Inp, Cols, Filters))
                        
                        # 算下一個XB的位置
                        xb_n += 1
                        if xb_n >= HW().Xbar_num:
                            xb_n = 0
                            cu_n += 1
                            if cu_n >= HW().CU_num:
                                cu_n = 0
                                pe_w += 1
                                used_pe_num += 1
                                if pe_w >= HW().PE_num_x:
                                    pe_w = 0
                                    pe_h += 1
                                    if pe_h >= HW().PE_num_y:
                                        pe_h = 0
                                        if rt_h % 2 == 0:
                                            rt_w += 1
                                            if rt_w >= HW().Router_num_x:
                                                rt_w -= 1
                                                rt_h += 1
                                                if rt_h >= HW().Router_num_y:
                                                    rt_h, rt_w = 0, 0
                                                    # print("no enough crossbar")
                                                    # exit()
                                        else:
                                            rt_w -= 1
                                            if rt_w < 0:
                                                rt_w = 0
                                                rt_h += 1
                                                if rt_h >= HW().Router_num_y:
                                                    rt_h, rt_w = 0, 0
                                                    # print("no enough crossbar")
                                                    # exit()
                                self.layer_used_pe[nlayer].append((rt_h, rt_w, pe_h, pe_w))

                # next layer next PE
                if cu_n != 0 or xb_n != 0:
                    pe_w += 1
                    if pe_w >= HW().PE_num_x:
                        pe_w = 0
                        pe_h += 1
                        if pe_h >= HW().PE_num_y:
                            pe_h = 0
                            if rt_h % 2 == 0:
                                rt_w += 1
                                if rt_w >= HW().Router_num_x:
                                    rt_w -= 1
                                    rt_h += 1
                                    if rt_h >= HW().Router_num_y:
                                        rt_h, rt_w = 0, 0
                                        # print("no enough crossbar")
                                        # exit()
                            else:
                                rt_w -= 1
                                if rt_w < 0:
                                    rt_w = 0
                                    rt_h += 1
                                    if rt_h >= HW().Router_num_y:
                                        rt_h, rt_w = 0, 0
                                        # print("no enough crossbar")
                                        # exit()
                else:
                    used_pe_num -= 1
                    self.layer_used_pe[nlayer].remove((rt_h, rt_w, pe_h, pe_w))

            elif self.model_info.layer_list[nlayer].layer_type == "pooling":
                print("Pooling", nlayer)
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
                self.layer_used_pe[nlayer].append((rt_h, rt_w, pe_h, pe_w))

                rt_h, rt_w, pe_h, pe_w = next_layer_id[0], next_layer_id[1], next_layer_id[2], next_layer_id[3]

    def __str__(self):
        return str(self.__dict__)
