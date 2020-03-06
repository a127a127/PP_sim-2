from HardwareMetaData import HardwareMetaData as HW
from ModelConfig import ModelConfig
from Model import Model
from MappingMetaData import MappingMetaData
import sys
from math import ceil
import numpy as np

class SameColumnFirstMapping(object):
    def __init__(self):
        # 規則: 同一筆data要放在同一個crossbar
        model_config = ModelConfig()
        print("Model:", model_config.Model_type)
        self.model_info = Model(model_config)

        self.layer_mapping_to_xbar = []
        self.layer_mapping_to_pe = []
        for rty_idx in range(HW().Router_num_y):
            self.layer_mapping_to_xbar.append([])
            self.layer_mapping_to_pe.append([])
            for rtx_idx in range(HW().Router_num_x):
                self.layer_mapping_to_xbar[rty_idx].append([])
                self.layer_mapping_to_pe[rty_idx].append([])
                for pey_idx in range(HW().PE_num_y):
                    self.layer_mapping_to_xbar[rty_idx][rtx_idx].append([])
                    self.layer_mapping_to_pe[rty_idx][rtx_idx].append([])
                    for pex_idx in range(HW().PE_num_x):
                        self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx].append([])
                        self.layer_mapping_to_pe[rty_idx][rtx_idx][pey_idx].append([])
                        for nlayer in range(self.model_info.layer_length):
                            self.layer_mapping_to_pe[rty_idx][rtx_idx][pey_idx][pex_idx].append([])
                        for cuy_idx in range(HW().CU_num_y):
                            self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx].append([])
                            for cux_idx in range(HW().CU_num_x):
                                self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx].append([])
                                for xby_idx in range(HW().Xbar_num_y):
                                    self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx].append([])
                                    for xbx_idx in range(HW().Xbar_num_x):
                                        self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx].append([])
                                        for nlayer in range(self.model_info.layer_length):
                                            self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx][xbx_idx].append([])

        self.map()

    def map(self):
        rt_h, rt_w = 0, 0
        pe_h, pe_w = 0, 0
        cu_h, cu_w = 0, 0
        xb_h, xb_w = 0, 0
        
        for nlayer in range(self.model_info.layer_length):
            if self.model_info.layer_list[nlayer].layer_type == "convolution":
                print("Convolution", nlayer)
                # for pooling
                pool_pe_id = (rt_h, rt_w, pe_h, pe_w)
                used_pe_num = 1

                # Inputs
                strides = self.model_info.strides[nlayer]
                inputs = []
                o_height = self.model_info.input_h[nlayer+1]
                o_width = self.model_info.input_w[nlayer+1]
                for oh in range(o_height):
                    for ow in range(o_width):
                        num_input = oh * o_width + ow
                        nn = []
                        for c in range(self.model_info.filter_c[nlayer]):
                            for h in range(self.model_info.filter_h[nlayer]):
                                for w in range(self.model_info.filter_w[nlayer]):
                                    nn.append([num_input, oh*strides+h, ow*strides+w, c]) # input feature map position
                        inputs.append(nn)
                inputs = np.array(inputs)

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
                        inp = inputs[:, h * HW().Xbar_h : (h+1) * HW().Xbar_h].tolist()
                        for Inp in inp:
                            self.layer_mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w][nlayer].append(MappingMetaData(Inp, Cols, Filters))
                        
                        # 算下一個XB的位置
                        xb_w += 1
                        if xb_w >= HW().Xbar_num_x:
                            xb_w = 0
                            xb_h += 1
                            if xb_h >= HW().Xbar_num_y:
                                xb_h = 0
                                cu_w += 1
                                if cu_w >= HW().CU_num_x:
                                    cu_w = 0
                                    cu_h += 1
                                    if cu_h >= HW().CU_num_y:
                                        cu_h = 0
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
                
                # next layer next PE
                if cu_h != 0 or  cu_w != 0 or xb_h != 0 or xb_w != 0:
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
        
            elif self.model_info.layer_list[nlayer].layer_type == "fully":
                print("Fully", nlayer)
                ## Inputs
                nn = []
                for h in range(self.model_info.filter_length[nlayer]):
                    nn.append([0, h, 0, 0])
                inputs = [nn]
                inputs = np.array(inputs)

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
                        inp = inputs[:, h * HW().Xbar_h : (h+1) * HW().Xbar_h].tolist()
                        for Inp in inp:
                            self.layer_mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w][nlayer].append(MappingMetaData(Inp, Cols, Filters))
                        
                        # 算下一個XB的位置
                        xb_w += 1
                        if xb_w >= HW().Xbar_num_x:
                            xb_w = 0
                            xb_h += 1
                            if xb_h >= HW().Xbar_num_y:
                                xb_h = 0
                                cu_w += 1
                                if cu_w >= HW().CU_num_x:
                                    cu_w = 0
                                    cu_h += 1
                                    if cu_h >= HW().CU_num_y:
                                        cu_h = 0
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

                # next layer next PE
                if cu_h != 0 or  cu_w != 0 or xb_h != 0 or xb_w != 0:
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

            elif self.model_info.layer_list[nlayer].layer_type == "pooling":
                print("Pooling", nlayer)
                o_height = self.model_info.input_h[nlayer+1]
                o_width = self.model_info.input_w[nlayer+1]
                inputs = []
                for oh in range(o_height):
                    for ow in range(o_width):
                        for c in range(self.model_info.input_c[nlayer]):
                            num_input = oh * o_width + ow + c * o_height * o_width
                            nn = []
                            for ph in range(self.model_info.pooling_h[nlayer]):
                                for pw in range(self.model_info.pooling_w[nlayer]):
                                    # nn.append([num_input,
                                    #            oh * self.model_info.pooling_strides[nlayer] + ph,
                                    #            ow * self.model_info.pooling_strides[nlayer] + pw,
                                    #            c])
                                    nn.append([num_input, nlayer, 
                                               oh * self.model_info.pooling_strides[nlayer] + ph,
                                               ow * self.model_info.pooling_strides[nlayer] + pw,
                                               c])
                            inputs.append(nn)
                inputs = np.array(inputs)

                input_per_pe = len(inputs) // used_pe_num # split into multiple pe
                if input_per_pe == 0:
                    input_per_pe = 1
                
                next_layer_id = (rt_h, rt_w, pe_h, pe_w)
                rt_h, rt_w, pe_h, pe_w = pool_pe_id[0], pool_pe_id[1], pool_pe_id[2], pool_pe_id[3]
                for pe_n in range(used_pe_num):
                    if pe_n != 0:
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
                                else:
                                    rt_w -= 1
                                    if rt_w < 0:
                                        rt_w = 0
                                        rt_h += 1

                    if pe_n + 1 == used_pe_num:
                        this_input = inputs[pe_n * input_per_pe : ].tolist()
                    else:
                        this_input = inputs[pe_n * input_per_pe : (pe_n+1) * input_per_pe].tolist()

                    if this_input:
                        self.layer_mapping_to_pe[rt_h][rt_w][pe_h][pe_w][nlayer].append(this_input)

                    rt_h, rt_w, pe_h, pe_w = next_layer_id[0], next_layer_id[1], next_layer_id[2], next_layer_id[3]


    def __str__(self):
        return str(self.__dict__)

class SameRowFirstMapping(object):
    def __init__(self):
        # 規則: 同一筆data要放在同一個crossbar
        model_config = ModelConfig()
        print("Model:", model_config.Model_type)
        self.model_info = Model(model_config)

        self.layer_mapping_to_xbar = []
        self.layer_mapping_to_pe = []
        for rty_idx in range(HW().Router_num_y):
            self.layer_mapping_to_xbar.append([])
            self.layer_mapping_to_pe.append([])
            for rtx_idx in range(HW().Router_num_x):
                self.layer_mapping_to_xbar[rty_idx].append([])
                self.layer_mapping_to_pe[rty_idx].append([])
                for pey_idx in range(HW().PE_num_y):
                    self.layer_mapping_to_xbar[rty_idx][rtx_idx].append([])
                    self.layer_mapping_to_pe[rty_idx][rtx_idx].append([])
                    for pex_idx in range(HW().PE_num_x):
                        self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx].append([])
                        self.layer_mapping_to_pe[rty_idx][rtx_idx][pey_idx].append([])
                        for nlayer in range(self.model_info.layer_length):
                            self.layer_mapping_to_pe[rty_idx][rtx_idx][pey_idx][pex_idx].append([])
                        for cuy_idx in range(HW().CU_num_y):
                            self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx].append([])
                            for cux_idx in range(HW().CU_num_x):
                                self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx].append([])
                                for xby_idx in range(HW().Xbar_num_y):
                                    self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx].append([])
                                    for xbx_idx in range(HW().Xbar_num_x):
                                        self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx].append([])
                                        for nlayer in range(self.model_info.layer_length):
                                            self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx][xbx_idx].append([])

        self.map()

    def map(self):
        rt_h, rt_w = 0, 0
        pe_h, pe_w = 0, 0
        cu_h, cu_w = 0, 0
        xb_h, xb_w = 0, 0
        
        for nlayer in range(self.model_info.layer_length):
            if self.model_info.layer_list[nlayer].layer_type == "convolution":
                print("Convolution", nlayer)
                # for pooling
                pool_pe_id = (rt_h, rt_w, pe_h, pe_w)
                used_pe_num = 1

                # Inputs
                strides = self.model_info.strides[nlayer]
                inputs = []
                o_height = self.model_info.input_h[nlayer+1]
                o_width = self.model_info.input_w[nlayer+1]
                for oh in range(o_height):
                    for ow in range(o_width):
                        num_input = oh * o_width + ow
                        nn = []
                        for c in range(self.model_info.filter_c[nlayer]):
                            for h in range(self.model_info.filter_h[nlayer]):
                                for w in range(self.model_info.filter_w[nlayer]):
                                    nn.append([num_input, oh*strides+h, ow*strides+w, c]) # input feature map position
                        inputs.append(nn)
                inputs = np.array(inputs)

                ## Weight matrix
                cells_per_weight = ceil(self.model_info.filter_bit / HW().cell_bit_width) # 16/2 = 8 cells per weight
                matrix_height = self.model_info.filter_length[nlayer]
                matrix_width  = self.model_info.filter_n[nlayer]
                filters_per_xb = HW().Xbar_w // cells_per_weight

                mapping_height_num_xb = ceil(matrix_height / HW().Xbar_h)
                mapping_width_num_xb  = ceil(matrix_width  / filters_per_xb) # width / 16
                for h in range(mapping_height_num_xb):
                    inp = inputs[:, h * HW().Xbar_h : (h+1) * HW().Xbar_h].tolist()
                    for w in range(mapping_width_num_xb):
                        # 一次map一個xb
                        # Filters
                        if w + 1 == mapping_width_num_xb:
                            Filters = [i for i in range(w * filters_per_xb, matrix_width)]
                        else:
                            Filters = [i for i in range(w * filters_per_xb,(w+1) * filters_per_xb)]
                        Cols = len(Filters) * cells_per_weight
                        for Inp in inp:
                            self.layer_mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w][nlayer].append(MappingMetaData(Inp, Cols, Filters))

                        # 算下一個XB的位置
                        xb_w += 1
                        if xb_w >= HW().Xbar_num_x:
                            xb_w = 0
                            xb_h += 1
                            if xb_h >= HW().Xbar_num_y:
                                xb_h = 0
                                cu_w += 1
                                if cu_w >= HW().CU_num_x:
                                    cu_w = 0
                                    cu_h += 1
                                    if cu_h >= HW().CU_num_y:
                                        cu_h = 0
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
                
                # next layer next PE
                if cu_h != 0 or  cu_w != 0 or xb_h != 0 or xb_w != 0:
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
        
            elif self.model_info.layer_list[nlayer].layer_type == "fully":
                print("Fully", nlayer)
                ## Inputs
                nn = []
                for h in range(self.model_info.filter_length[nlayer]):
                    nn.append([0, h, 0, 0])
                inputs = [nn]
                inputs = np.array(inputs)

                ## Weight matrix
                cells_per_weight = ceil(self.model_info.filter_bit / HW().cell_bit_width) # 16/2 = 8 cells per weight
                matrix_height = self.model_info.filter_length[nlayer]
                matrix_width  = self.model_info.filter_n[nlayer]
                filters_per_xb = HW().Xbar_w // cells_per_weight

                mapping_height_num_xb = ceil(matrix_height / HW().Xbar_h)
                mapping_width_num_xb  = ceil(matrix_width  / filters_per_xb) # width / 16
                for h in range(mapping_height_num_xb):
                    inp = inputs[:, h * HW().Xbar_h : (h+1) * HW().Xbar_h].tolist()
                    for w in range(mapping_width_num_xb):
                        # 一次map一個xb
                        # Filters
                        if w + 1 == mapping_width_num_xb:
                            Filters = [i for i in range(w * filters_per_xb, matrix_width)]
                        else:
                            Filters = [i for i in range(w * filters_per_xb,(w+1) * filters_per_xb)]
                        Cols = len(Filters) * cells_per_weight
                        for Inp in inp:
                            self.layer_mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w][nlayer].append(MappingMetaData(Inp, Cols, Filters))

                        # 算下一個XB的位置
                        xb_w += 1
                        if xb_w >= HW().Xbar_num_x:
                            xb_w = 0
                            xb_h += 1
                            if xb_h >= HW().Xbar_num_y:
                                xb_h = 0
                                cu_w += 1
                                if cu_w >= HW().CU_num_x:
                                    cu_w = 0
                                    cu_h += 1
                                    if cu_h >= HW().CU_num_y:
                                        cu_h = 0
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
                                                        
                
                # next layer next PE
                if cu_h != 0 or  cu_w != 0 or xb_h != 0 or xb_w != 0:
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

            elif self.model_info.layer_list[nlayer].layer_type == "pooling":
                print("Pooling", nlayer)
                o_height = self.model_info.input_h[nlayer+1]
                o_width = self.model_info.input_w[nlayer+1]
                inputs = []
                for oh in range(o_height):
                    for ow in range(o_width):
                        for c in range(self.model_info.input_c[nlayer]):
                            num_input = oh * o_width + ow + c * o_height * o_width
                            nn = []
                            for ph in range(self.model_info.pooling_h[nlayer]):
                                for pw in range(self.model_info.pooling_w[nlayer]):
                                    # nn.append([num_input,
                                    #            oh * self.model_info.pooling_strides[nlayer] + ph,
                                    #            ow * self.model_info.pooling_strides[nlayer] + pw,
                                    #            c])
                                    nn.append([num_input, nlayer,
                                               oh * self.model_info.pooling_strides[nlayer] + ph,
                                               ow * self.model_info.pooling_strides[nlayer] + pw,
                                               c])
                            inputs.append(nn)
                inputs = np.array(inputs)

                input_per_pe = len(inputs) // used_pe_num # split into multiple pe
                if input_per_pe == 0:
                    input_per_pe = 1
                
                next_layer_id = (rt_h, rt_w, pe_h, pe_w)
                rt_h, rt_w, pe_h, pe_w = pool_pe_id[0], pool_pe_id[1], pool_pe_id[2], pool_pe_id[3]
                for pe_n in range(used_pe_num):
                    if pe_n != 0:
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
                                else:
                                    rt_w -= 1
                                    if rt_w < 0:
                                        rt_w = 0
                                        rt_h += 1

                    if pe_n + 1 == used_pe_num:
                        this_input = inputs[pe_n * input_per_pe : ].tolist()
                    else:
                        this_input = inputs[pe_n * input_per_pe : (pe_n+1) * input_per_pe].tolist()

                    if this_input:
                        self.layer_mapping_to_pe[rt_h][rt_w][pe_h][pe_w][nlayer].append(this_input)

                rt_h, rt_w, pe_h, pe_w = next_layer_id[0], next_layer_id[1], next_layer_id[2], next_layer_id[3]

    def __str__(self):
        return str(self.__dict__)

class ParallelsimMapping(object):
    def __init__(self, parall):
        self.Parall = parall
        model_config = ModelConfig()
        print("Model:", model_config.Model_type)
        self.model_info = Model(model_config)

        self.layer_mapping_to_xbar = []
        self.layer_mapping_to_pe = []
        for rty_idx in range(HW().Router_num_y):
            self.layer_mapping_to_xbar.append([])
            self.layer_mapping_to_pe.append([])
            for rtx_idx in range(HW().Router_num_x):
                self.layer_mapping_to_xbar[rty_idx].append([])
                self.layer_mapping_to_pe[rty_idx].append([])
                for pey_idx in range(HW().PE_num_y):
                    self.layer_mapping_to_xbar[rty_idx][rtx_idx].append([])
                    self.layer_mapping_to_pe[rty_idx][rtx_idx].append([])
                    for pex_idx in range(HW().PE_num_x):
                        self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx].append([])
                        self.layer_mapping_to_pe[rty_idx][rtx_idx][pey_idx].append([])
                        for nlayer in range(self.model_info.layer_length):
                            self.layer_mapping_to_pe[rty_idx][rtx_idx][pey_idx][pex_idx].append([])
                        for cuy_idx in range(HW().CU_num_y):
                            self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx].append([])
                            for cux_idx in range(HW().CU_num_x):
                                self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx].append([])
                                for xby_idx in range(HW().Xbar_num_y):
                                    self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx].append([])
                                    for xbx_idx in range(HW().Xbar_num_x):
                                        self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx].append([])
                                        for nlayer in range(self.model_info.layer_length):
                                            self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx][xbx_idx].append([])

        self.map()

    def map(self):
        rt_h, rt_w = 0, 0
        pe_h, pe_w = 0, 0
        cu_h, cu_w = 0, 0
        xb_h, xb_w = 0, 0
        
        for nlayer in range(self.model_info.layer_length):
            if self.model_info.layer_list[nlayer].layer_type == "convolution":
                print("Convolution", nlayer)
                # for pooling
                pool_pe_id = (rt_h, rt_w, pe_h, pe_w)
                used_pe_num = 1

                # Inputs
                strides = self.model_info.strides[nlayer]
                inputs = []
                o_height = self.model_info.input_h[nlayer+1]
                o_width = self.model_info.input_w[nlayer+1]
                for oh in range(o_height):
                    for ow in range(o_width):
                        num_input = oh * o_width + ow
                        nn = []
                        for c in range(self.model_info.filter_c[nlayer]):
                            for h in range(self.model_info.filter_h[nlayer]):
                                for w in range(self.model_info.filter_w[nlayer]):
                                    nn.append([num_input, oh*strides+h, ow*strides+w, c]) # input feature map position
                        inputs.append(nn)
                inputs = np.array(inputs)

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
                    inp = inputs[:, h * H : (h+1) * H].tolist() # 4
                    for w in range(mapping_width_num_xb):
                        # 一次map一個xb
                        # Filters
                        if w + 1 == mapping_width_num_xb:
                            Filters = [i for i in range(w * filters_per_xb, matrix_width)]
                        else:
                            Filters = [i for i in range(w * filters_per_xb,(w+1) * filters_per_xb)]
                        Cols = len(Filters) * cells_per_weight
                        for Inp in inp:
                            self.layer_mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w][nlayer].append(MappingMetaData(Inp, Cols, Filters))

                        # 算下一個XB的位置
                        xb_w += 1
                        if xb_w >= HW().Xbar_num_x:
                            xb_w = 0
                            xb_h += 1
                            if xb_h >= HW().Xbar_num_y:
                                xb_h = 0
                                cu_w += 1
                                if cu_w >= HW().CU_num_x:
                                    cu_w = 0
                                    cu_h += 1
                                    if cu_h >= HW().CU_num_y:
                                        cu_h = 0
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
                                                else:
                                                    rt_w -= 1
                                                    if rt_w < 0:
                                                        rt_w = 0
                                                        rt_h += 1
                                                        if rt_h >= HW().Router_num_y:
                                                            rt_h, rt_w = 0, 0
                # next layer next PE
                if cu_h != 0 or  cu_w != 0 or xb_h != 0 or xb_w != 0:
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
                            else:
                                rt_w -= 1
                                if rt_w < 0:
                                    rt_w = 0
                                    rt_h += 1
                                    if rt_h >= HW().Router_num_y:
                                        rt_h, rt_w = 0, 0
                else:
                    used_pe_num -= 1
        
            elif self.model_info.layer_list[nlayer].layer_type == "fully":
                print("Fully", nlayer)
                ## Inputs
                nn = []
                for h in range(self.model_info.filter_length[nlayer]):
                    nn.append([0, h, 0, 0])
                inputs = [nn]
                inputs = np.array(inputs)

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
                    inp = inputs[:, h * H : (h+1) * H].tolist() # 4
                    for w in range(mapping_width_num_xb):
                        # 一次map一個xb
                        # Filters
                        if w + 1 == mapping_width_num_xb:
                            Filters = [i for i in range(w * filters_per_xb, matrix_width)]
                        else:
                            Filters = [i for i in range(w * filters_per_xb,(w+1) * filters_per_xb)]
                        Cols = len(Filters) * cells_per_weight
                        for Inp in inp:
                            self.layer_mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w][nlayer].append(MappingMetaData(Inp, Cols, Filters))

                        # 算下一個XB的位置
                        xb_w += 1
                        if xb_w >= HW().Xbar_num_x:
                            xb_w = 0
                            xb_h += 1
                            if xb_h >= HW().Xbar_num_y:
                                xb_h = 0
                                cu_w += 1
                                if cu_w >= HW().CU_num_x:
                                    cu_w = 0
                                    cu_h += 1
                                    if cu_h >= HW().CU_num_y:
                                        cu_h = 0
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
                                                else:
                                                    rt_w -= 1
                                                    if rt_w < 0:
                                                        rt_w = 0
                                                        rt_h += 1
                                                        if rt_h >= HW().Router_num_y:
                                                            rt_h, rt_w = 0, 0
                                                        
                
                # next layer next PE
                if cu_h != 0 or  cu_w != 0 or xb_h != 0 or xb_w != 0:
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
                            else:
                                rt_w -= 1
                                if rt_w < 0:
                                    rt_w = 0
                                    rt_h += 1
                                    if rt_h >= HW().Router_num_y:
                                        rt_h, rt_w = 0, 0

            elif self.model_info.layer_list[nlayer].layer_type == "pooling":
                print("Pooling", nlayer)
                o_height = self.model_info.input_h[nlayer+1]
                o_width = self.model_info.input_w[nlayer+1]
                inputs = []
                for oh in range(o_height):
                    for ow in range(o_width):
                        for c in range(self.model_info.input_c[nlayer]):
                            num_input = oh * o_width + ow + c * o_height * o_width
                            nn = []
                            for ph in range(self.model_info.pooling_h[nlayer]):
                                for pw in range(self.model_info.pooling_w[nlayer]):
                                    # nn.append([num_input,
                                    #            oh * self.model_info.pooling_strides[nlayer] + ph,
                                    #            ow * self.model_info.pooling_strides[nlayer] + pw,
                                    #            c])
                                    nn.append([num_input, nlayer, 
                                               oh * self.model_info.pooling_strides[nlayer] + ph,
                                               ow * self.model_info.pooling_strides[nlayer] + pw,
                                               c])
                            inputs.append(nn)
                inputs = np.array(inputs)

                input_per_pe = len(inputs) // used_pe_num # split into multiple pe
                if input_per_pe == 0:
                    input_per_pe = 1
                
                next_layer_id = (rt_h, rt_w, pe_h, pe_w)
                rt_h, rt_w, pe_h, pe_w = pool_pe_id[0], pool_pe_id[1], pool_pe_id[2], pool_pe_id[3]
                for pe_n in range(used_pe_num):
                    if pe_n != 0:
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
                                else:
                                    rt_w -= 1
                                    if rt_w < 0:
                                        rt_w = 0
                                        rt_h += 1

                    if pe_n + 1 == used_pe_num:
                        this_input = inputs[pe_n * input_per_pe : ].tolist()
                    else:
                        this_input = inputs[pe_n * input_per_pe : (pe_n+1) * input_per_pe].tolist()

                    if this_input:
                        self.layer_mapping_to_pe[rt_h][rt_w][pe_h][pe_w][nlayer].append(this_input)

                rt_h, rt_w, pe_h, pe_w = next_layer_id[0], next_layer_id[1], next_layer_id[2], next_layer_id[3]

    def __str__(self):
        return str(self.__dict__)
