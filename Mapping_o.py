from Model import Model
from MappingMetaData import MappingMetaData
import sys, csv
from math import ceil
import numpy as np

# Lenet:8, Cifar10: 5, DeepID: 6, Caffenet: 321, Overfeat: 568, VGG16: 708
# Lenet: 1, 1, 0, 0
# Cifar10: 0, 1, 0, 1
# DeepID: 0, 1, 1, 0
# Caffenet: 8, 0, 0, 1
# Overfeat: 11, 1, 0, 0 
# VGG16: 12, 11, 0, 0

class SameColumnFirstMapping(object):
    def __init__(self, model_config, hw_config):
        # 同一筆data要放在同一個crossbar
        print("Model:", model_config.Model_type)
        self.model_info = Model(model_config)
        self.hw_config  = hw_config

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
        self.layer_used_pe = []
        self.layer_used_xb_num = []
        for nlayer in range(self.model_info.layer_length):
            self.layer_used_pe.append(set())
            self.layer_used_xb_num.append(0)

        if False:
            used_pe = 0
            for nlayer in range(len(self.layer_used_pe)):
                print("layer", nlayer, "use", len(self.layer_used_pe[nlayer]))
                used_pe += len(self.layer_used_pe[nlayer])
            print("used PE", used_pe)
            exit()
        
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

                self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))

                ## Weight matrix
                cells_per_weight = ceil(self.model_info.filter_bit / self.hw_config.cell_bit_width) # 16/2 = 8 cells per weight
                matrix_height = self.model_info.filter_length[nlayer]
                matrix_width  = self.model_info.filter_n[nlayer]
                filters_per_xb = self.hw_config.Xbar_w // cells_per_weight

                mapping_height_num_xb = ceil(matrix_height / self.hw_config.Xbar_h)
                mapping_width_num_xb  = ceil(matrix_width  / filters_per_xb) # width / 16
                for w in range(mapping_width_num_xb):
                    # Filters
                    if w + 1 == mapping_width_num_xb: # 邊界
                        Filters = [i for i in range(w * filters_per_xb, matrix_width)]
                    else:
                        Filters = [i for i in range(w * filters_per_xb,(w+1) * filters_per_xb)]
                    Cols = len(Filters) * cells_per_weight
                    
                    for h in range(mapping_height_num_xb):
                        self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))
                        # 一次map一個xb
                        if h != mapping_height_num_xb-1:
                            Inp = [i for i in range(h * self.hw_config.Xbar_h,(h+1) * self.hw_config.Xbar_h)]
                        else:
                            Inp = [i for i in range(h * self.hw_config.Xbar_h, matrix_height)]
                        self.mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_n][xb_n][nlayer].append(MappingMetaData(Inp, Cols, Filters))
                        
                        # 算下一個XB的位置
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
                                                    print("no enough crossbar")
                                                    exit()
                                        else:
                                            rt_w -= 1
                                            if rt_w < 0:
                                                rt_w = 0
                                                rt_h += 1
                                                if rt_h >= self.hw_config.Router_num_y:
                                                    print("no enough crossbar")
                                                    exit()
                
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
                                        print("no enough crossbar")
                                        exit()
                            else:
                                rt_w -= 1
                                if rt_w < 0:
                                    rt_w = 0
                                    rt_h += 1
                                    if rt_h >= self.hw_config.Router_num_y:
                                        print("no enough crossbar")
                                        exit()
                # else:
                #     if len(self.layer_used_pe[nlayer]) < self.hw_config.total_pe_num:
                #         self.layer_used_pe[nlayer].remove((rt_h, rt_w, pe_h, pe_w))
        
            elif self.model_info.layer_list[nlayer].layer_type == "fully":
                print("Fully", nlayer)
                self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))

                ## Weight matrix
                cells_per_weight = ceil(self.model_info.filter_bit / self.hw_config.cell_bit_width) # 16/2 = 8 cells per weight
                matrix_height = self.model_info.filter_length[nlayer]
                matrix_width  = self.model_info.filter_n[nlayer]
                filters_per_xb = self.hw_config.Xbar_w // cells_per_weight

                mapping_height_num_xb = ceil(matrix_height / self.hw_config.Xbar_h)
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
                        self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))
                        # 一次map一個xb
                        if h != mapping_height_num_xb-1:
                            Inp = [i for i in range(h * self.hw_config.Xbar_h,(h+1) * self.hw_config.Xbar_h)]
                        else:
                            Inp = [i for i in range(h * self.hw_config.Xbar_h, matrix_height)]
                        self.mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_n][xb_n][nlayer].append(MappingMetaData(Inp, Cols, Filters))
                        
                        # 算下一個XB的位置
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
                                                    print("no enough crossbar")
                                                    exit()
                                        else:
                                            rt_w -= 1
                                            if rt_w < 0:
                                                rt_w = 0
                                                rt_h += 1
                                                if rt_h >= self.hw_config.Router_num_y:
                                                    print("no enough crossbar")
                                                    exit()  
                
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
                                        print("no enough crossbar")
                                        exit()
                            else:
                                rt_w -= 1
                                if rt_w < 0:
                                    rt_w = 0
                                    rt_h += 1
                                    if rt_h >= self.hw_config.Router_num_y:
                                        print("no enough crossbar")
                                        exit()
                # else:
                #     if len(self.layer_used_pe[nlayer]) < self.hw_config.total_pe_num:
                #         self.layer_used_pe[nlayer].remove((rt_h, rt_w, pe_h, pe_w))
                    
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

class SameRowFirstMapping(object):
    def __init__(self, model_config, hw_config):
        # 同一筆data要放在同一個crossbar
        print("Model:", model_config.Model_type)
        self.model_info = Model(model_config)
        self.hw_config = hw_config

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
        self.layer_used_pe = []
        self.layer_used_xb_num = []
        for nlayer in range(self.model_info.layer_length):
            self.layer_used_pe.append(set())
            self.layer_used_xb_num.append(0)
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

                self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))

                ## Weight matrix
                cells_per_weight = ceil(self.model_info.filter_bit / self.hw_config.cell_bit_width) # 16/2 = 8 cells per weight
                matrix_height = self.model_info.filter_length[nlayer]
                matrix_width  = self.model_info.filter_n[nlayer]
                filters_per_xb = self.hw_config.Xbar_w // cells_per_weight

                mapping_height_num_xb = ceil(matrix_height / self.hw_config.Xbar_h)
                mapping_width_num_xb  = ceil(matrix_width  / filters_per_xb) # width / 16
                
                for h in range(mapping_height_num_xb):
                    if h != mapping_height_num_xb-1:
                        Inp = [i for i in range(h * self.hw_config.Xbar_h,(h+1) * self.hw_config.Xbar_h)]
                    else:
                        Inp = [i for i in range(h * self.hw_config.Xbar_h, matrix_height)]
                    for w in range(mapping_width_num_xb):
                        self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))
                        # 一次map一個xb
                        if w + 1 == mapping_width_num_xb:
                            Filters = [i for i in range(w * filters_per_xb, matrix_width)]
                        else:
                            Filters = [i for i in range(w * filters_per_xb,(w+1) * filters_per_xb)]
                        Cols = len(Filters) * cells_per_weight
                        self.mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_n][xb_n][nlayer].append(MappingMetaData(Inp, Cols, Filters))
                        
                        # 算下一個XB的位置
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
                                                    print("no enough crossbar")
                                                    exit()
                                        else:
                                            rt_w -= 1
                                            if rt_w < 0:
                                                rt_w = 0
                                                rt_h += 1
                                                if rt_h >= self.hw_config.Router_num_y:
                                                    print("no enough crossbar")
                                                    exit()
                
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
                                        print("no enough crossbar")
                                        exit()
                            else:
                                rt_w -= 1
                                if rt_w < 0:
                                    rt_w = 0
                                    rt_h += 1
                                    if rt_h >= self.hw_config.Router_num_y:
                                        print("no enough crossbar")
                                        exit()
                # else:
                #     if len(self.layer_used_pe[nlayer]) < self.hw_config.total_pe_num:
                #         self.layer_used_pe[nlayer].remove((rt_h, rt_w, pe_h, pe_w))
        
            elif self.model_info.layer_list[nlayer].layer_type == "fully":
                print("Fully", nlayer)
                self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))

                ## Weight matrix
                cells_per_weight = ceil(self.model_info.filter_bit / self.hw_config.cell_bit_width) # 16/2 = 8 cells per weight
                matrix_height = self.model_info.filter_length[nlayer]
                matrix_width  = self.model_info.filter_n[nlayer]
                filters_per_xb = self.hw_config.Xbar_w // cells_per_weight

                mapping_height_num_xb = ceil(matrix_height / self.hw_config.Xbar_h)
                mapping_width_num_xb  = ceil(matrix_width  / filters_per_xb) # width / 16
                for h in range(mapping_height_num_xb):
                    if h != mapping_height_num_xb-1:
                        Inp = [i for i in range(h * self.hw_config.Xbar_h,(h+1) * self.hw_config.Xbar_h)]
                    else:
                        Inp = [i for i in range(h * self.hw_config.Xbar_h, matrix_height)]
                    for w in range(mapping_width_num_xb):
                        self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))
                        # 一次map一個xb
                        if w + 1 == mapping_width_num_xb:
                            Filters = [i for i in range(w * filters_per_xb, matrix_width)]
                        else:
                            Filters = [i for i in range(w * filters_per_xb,(w+1) * filters_per_xb)]
                        Cols = len(Filters) * cells_per_weight
                        self.mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_n][xb_n][nlayer].append(MappingMetaData(Inp, Cols, Filters))
                        
                        # 算下一個XB的位置
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
                                                    print("no enough crossbar")
                                                    exit()
                                        else:
                                            rt_w -= 1
                                            if rt_w < 0:
                                                rt_w = 0
                                                rt_h += 1
                                                if rt_h >= self.hw_config.Router_num_y:
                                                    print("no enough crossbar")
                                                    exit()
                
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
                                        print("no enough crossbar")
                                        exit()
                            else:
                                rt_w -= 1
                                if rt_w < 0:
                                    rt_w = 0
                                    rt_h += 1
                                    if rt_h >= self.hw_config.Router_num_y:
                                        print("no enough crossbar")
                                        exit()
                # else:
                #     if len(self.layer_used_pe[nlayer]) < self.hw_config.total_pe_num:
                #         self.layer_used_pe[nlayer].remove((rt_h, rt_w, pe_h, pe_w))
                
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

class SCFParallelsimMapping(object):
    def __init__(self, model_config, hw_config, parall):
        self.Parall = parall
        print("Model:", model_config.Model_type)
        if model_config.Model_type == "Lenet":
            RTY, RTX, PEY, PEX = 1, 1, 0, 0
        elif model_config.Model_type == "Cifar10":
            RTY, RTX, PEY, PEX = 0, 1, 0, 1
        elif model_config.Model_type == "DeepID":
            RTY, RTX, PEY, PEX = 0, 1, 1, 0
        elif model_config.Model_type == "Caffenet":
            RTY, RTX, PEY, PEX = 8, 0, 0, 1
        elif model_config.Model_type == "Overfeat":
            RTY, RTX, PEY, PEX = 11, 1, 0, 0 
        
        self.model_info = Model(model_config)
        self.hw_config = hw_config

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
        self.layer_used_pe = []
        self.layer_used_xb_num = []
        for nlayer in range(self.model_info.layer_length):
            self.layer_used_pe.append(set())
            self.layer_used_xb_num.append(0)
        self.map(RTY, RTX, PEY, PEX)

    def map(self, RTY, RTX, PEY, PEX):
        rt_h, rt_w = 0, 0
        pe_h, pe_w = 0, 0
        cu_n = 0
        xb_n = 0
        
        for nlayer in range(self.model_info.layer_length):
            if self.model_info.layer_list[nlayer].layer_type == "convolution":
                print("Convolution", nlayer)
                # for pooling
                pool_pe_id = (rt_h, rt_w, pe_h, pe_w)

                self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))
                
                ## Weight matrix
                cells_per_weight = ceil(self.model_info.filter_bit / self.hw_config.cell_bit_width) # 16/2 = 8 cells per weight
                matrix_height = self.model_info.filter_length[nlayer]
                matrix_width  = self.model_info.filter_n[nlayer]
                filters_per_xb = self.hw_config.Xbar_w // cells_per_weight
                filters_per_xb = ceil(filters_per_xb / self.Parall) # 1

                H = ceil(self.hw_config.Xbar_h / self.Parall) # 2
                mapping_height_num_xb = ceil(matrix_height / H) # 3
                mapping_width_num_xb  = ceil(matrix_width  / filters_per_xb) # width / 16
                for w in range(mapping_width_num_xb):
                    if w + 1 == mapping_width_num_xb:
                        Filters = [i for i in range(w * filters_per_xb, matrix_width)]
                    else:
                        Filters = [i for i in range(w * filters_per_xb,(w+1) * filters_per_xb)]
                    Cols = len(Filters) * cells_per_weight
                    for h in range(mapping_height_num_xb):
                        self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))
                        # 一次map一個xb
                        if h != mapping_height_num_xb-1:
                            Inp = [i for i in range(h * H,(h+1) * H)]
                        else:
                            Inp = [i for i in range(h * H, matrix_height)]
                        self.mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_n][xb_n][nlayer].append(MappingMetaData(Inp, Cols, Filters))
                        
                        # 算下一個XB的位置
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
                                                    #print("no enough crossbar")
                                                    #exit()
                                        else:
                                            rt_w -= 1
                                            if rt_w < 0:
                                                rt_w = 0
                                                rt_h += 1
                                                if rt_h >= self.hw_config.Router_num_y:
                                                    rt_h, rt_w = 0, 0
                                                    #print("no enough crossbar")
                                                    #exit()
                                if rt_h == RTY and rt_w == RTX and pe_h == PEY and pe_w == PEX:
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
                                        # print("no enough crossbar")
                                        # exit()
                            else:
                                rt_w -= 1
                                if rt_w < 0:
                                    rt_w = 0
                                    rt_h += 1
                                    if rt_h >= self.hw_config.Router_num_y:
                                        rt_h, rt_w = 0, 0
                                        # print("no enough crossbar")
                                        # exit()
                    if rt_h == RTY and rt_w == RTX and pe_h == PEY and pe_w == PEX:
                        rt_h, rt_w, pe_h, pe_w = 0, 0, 0, 0
                # else:
                    # if len(self.layer_used_pe[nlayer]) < self.hw_config.total_pe_num:
                    #     self.layer_used_pe[nlayer].remove((rt_h, rt_w, pe_h, pe_w))

            elif self.model_info.layer_list[nlayer].layer_type == "fully":
                print("Fully", nlayer)
                self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))

                ## Weight matrix
                cells_per_weight = ceil(self.model_info.filter_bit / self.hw_config.cell_bit_width) # 16/2 = 8 cells per weight
                matrix_height = self.model_info.filter_length[nlayer]
                matrix_width  = self.model_info.filter_n[nlayer]
                filters_per_xb = self.hw_config.Xbar_w // cells_per_weight
                filters_per_xb = ceil(filters_per_xb / self.Parall) # 1

                H = ceil(self.hw_config.Xbar_h / self.Parall) # 2
                mapping_height_num_xb = ceil(matrix_height / H) # 3
                mapping_width_num_xb  = ceil(matrix_width  / filters_per_xb) # width / 16
                for w in range(mapping_width_num_xb):
                    if w + 1 == mapping_width_num_xb:
                        Filters = [i for i in range(w * filters_per_xb, matrix_width)]
                    else:
                        Filters = [i for i in range(w * filters_per_xb,(w+1) * filters_per_xb)]
                    Cols = len(Filters) * cells_per_weight
                    for h in range(mapping_height_num_xb):
                        self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))
                        # 一次map一個xb
                        if h != mapping_height_num_xb-1:
                            Inp = [i for i in range(h * H,(h+1) * H)]
                        else:
                            Inp = [i for i in range(h * H, matrix_height)]
                        self.mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_n][xb_n][nlayer].append(MappingMetaData(Inp, Cols, Filters))
                        
                        # 算下一個XB的位置
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
                                                    # print("no enough crossbar")
                                                    # exit()
                                        else:
                                            rt_w -= 1
                                            if rt_w < 0:
                                                rt_w = 0
                                                rt_h += 1
                                                if rt_h >= self.hw_config.Router_num_y:
                                                    rt_h, rt_w = 0, 0
                                                    # print("no enough crossbar")
                                                    # exit()
                                if rt_h == RTY and rt_w == RTX and pe_h == PEY and pe_w == PEX:
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
                                        # print("no enough crossbar")
                                        # exit()
                            else:
                                rt_w -= 1
                                if rt_w < 0:
                                    rt_w = 0
                                    rt_h += 1
                                    if rt_h >= self.hw_config.Router_num_y:
                                        rt_h, rt_w = 0, 0
                                        # print("no enough crossbar")
                                        # exit()
                    if rt_h == RTY and rt_w == RTX and pe_h == PEY and pe_w == PEX:
                        rt_h, rt_w, pe_h, pe_w = 0, 0, 0, 0
                # else:
                #     if len(self.layer_used_pe[nlayer]) < self.hw_config.total_pe_num:
                #         self.layer_used_pe[nlayer].remove((rt_h, rt_w, pe_h, pe_w))
                 
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

class SRFParallelsimMapping(object):
    def __init__(self, model_config, hw_config, parall):
        self.Parall = parall
        print("Model:", model_config.Model_type)
        if model_config.Model_type == "Lenet":
            RTY, RTX, PEY, PEX = 1, 1, 0, 0
        elif model_config.Model_type == "Cifar10":
            RTY, RTX, PEY, PEX = 0, 1, 0, 1
        elif model_config.Model_type == "DeepID":
            RTY, RTX, PEY, PEX = 0, 1, 1, 0
        elif model_config.Model_type == "Caffenet":
            RTY, RTX, PEY, PEX = 8, 0, 0, 1
        elif model_config.Model_type == "Overfeat":
            RTY, RTX, PEY, PEX = 11, 1, 0, 0 

        self.model_info = Model(model_config)
        self.hw_config  = hw_config

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
        self.layer_used_pe = []
        self.layer_used_xb_num = []
        for nlayer in range(self.model_info.layer_length):
            self.layer_used_pe.append(set())
            self.layer_used_xb_num.append(0)
        self.map(RTY, RTX, PEY, PEX)

    def map(self, RTY, RTX, PEY, PEX):
        rt_h, rt_w = 0, 0
        pe_h, pe_w = 0, 0
        cu_n = 0
        xb_n = 0
        
        for nlayer in range(self.model_info.layer_length):
            if self.model_info.layer_list[nlayer].layer_type == "convolution":
                print("Convolution", nlayer)
                # for pooling
                pool_pe_id = (rt_h, rt_w, pe_h, pe_w)

                self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))

                ## Weight matrix
                cells_per_weight = ceil(self.model_info.filter_bit / self.hw_config.cell_bit_width) # 16/2 = 8 cells per weight
                matrix_height = self.model_info.filter_length[nlayer]
                matrix_width  = self.model_info.filter_n[nlayer]
                filters_per_xb = self.hw_config.Xbar_w // cells_per_weight
                filters_per_xb = ceil(filters_per_xb / self.Parall) # 1

                H = ceil(self.hw_config.Xbar_h / self.Parall) # 2
                mapping_height_num_xb = ceil(matrix_height / H) # 3
                mapping_width_num_xb  = ceil(matrix_width  / filters_per_xb) # width / 16
                for h in range(mapping_height_num_xb):
                    if h != mapping_height_num_xb-1:
                        Inp = [i for i in range(h * H,(h+1) * H)]
                    else:
                        Inp = [i for i in range(h * H, matrix_height)]
                    for w in range(mapping_width_num_xb):
                        self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))
                        # 一次map一個xb
                        if w + 1 == mapping_width_num_xb:
                            Filters = [i for i in range(w * filters_per_xb, matrix_width)]
                        else:
                            Filters = [i for i in range(w * filters_per_xb,(w+1) * filters_per_xb)]
                        Cols = len(Filters) * cells_per_weight
                        self.mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_n][xb_n][nlayer].append(MappingMetaData(Inp, Cols, Filters))
                        
                        # 算下一個XB的位置
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
                                                    # print("no enough crossbar")
                                                    # exit()
                                        else:
                                            rt_w -= 1
                                            if rt_w < 0:
                                                rt_w = 0
                                                rt_h += 1
                                                if rt_h >= self.hw_config.Router_num_y:
                                                    rt_h, rt_w = 0, 0
                                                    # print("no enough crossbar")
                                                    # exit()
                                if rt_h == RTY and rt_w == RTX and pe_h == PEY and pe_w == PEX:
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
                                        # print("no enough crossbar")
                                        # exit()
                            else:
                                rt_w -= 1
                                if rt_w < 0:
                                    rt_w = 0
                                    rt_h += 1
                                    if rt_h >= self.hw_config.Router_num_y:
                                        rt_h, rt_w = 0, 0
                                        # print("no enough crossbar")
                                        # exit()
                    if rt_h == RTY and rt_w == RTX and pe_h == PEY and pe_w == PEX:
                        rt_h, rt_w, pe_h, pe_w = 0, 0, 0, 0
                # else:
                #     if len(self.layer_used_pe[nlayer]) < self.hw_config.total_pe_num:
                #         self.layer_used_pe[nlayer].remove((rt_h, rt_w, pe_h, pe_w))

            elif self.model_info.layer_list[nlayer].layer_type == "fully":
                print("Fully", nlayer)
                self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))

                ## Weight matrix
                cells_per_weight = ceil(self.model_info.filter_bit / self.hw_config.cell_bit_width) # 16/2 = 8 cells per weight
                matrix_height = self.model_info.filter_length[nlayer]
                matrix_width  = self.model_info.filter_n[nlayer]
                filters_per_xb = self.hw_config.Xbar_w // cells_per_weight
                filters_per_xb = ceil(filters_per_xb / self.Parall) # 1

                H = ceil(self.hw_config.Xbar_h / self.Parall) # 2
                mapping_height_num_xb = ceil(matrix_height / H) # 3
                mapping_width_num_xb  = ceil(matrix_width  / filters_per_xb) # width / 16
                for h in range(mapping_height_num_xb):
                    if h != mapping_height_num_xb-1:
                        Inp = [i for i in range(h * H,(h+1) * H)]
                    else:
                        Inp = [i for i in range(h * H, matrix_height)]
                    for w in range(mapping_width_num_xb):
                        self.layer_used_pe[nlayer].add((rt_h, rt_w, pe_h, pe_w))
                        # 一次map一個xb
                        if w + 1 == mapping_width_num_xb:
                            Filters = [i for i in range(w * filters_per_xb, matrix_width)]
                        else:
                            Filters = [i for i in range(w * filters_per_xb,(w+1) * filters_per_xb)]
                        Cols = len(Filters) * cells_per_weight
                        self.mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_n][xb_n][nlayer].append(MappingMetaData(Inp, Cols, Filters))
                        
                        # 算下一個XB的位置
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
                                                    # print("no enough crossbar")
                                                    # exit()
                                        else:
                                            rt_w -= 1
                                            if rt_w < 0:
                                                rt_w = 0
                                                rt_h += 1
                                                if rt_h >= self.hw_config.Router_num_y:
                                                    rt_h, rt_w = 0, 0
                                                    # print("no enough crossbar")
                                                    # exit()
                                
                                if rt_h == RTY and rt_w == RTX and pe_h == PEY and pe_w == PEX:
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
                                        # print("no enough crossbar")
                                        # exit()
                            else:
                                rt_w -= 1
                                if rt_w < 0:
                                    rt_w = 0
                                    rt_h += 1
                                    if rt_h >= self.hw_config.Router_num_y:
                                        rt_h, rt_w = 0, 0
                                        # print("no enough crossbar")
                                        # exit()
                    if rt_h == RTY and rt_w == RTX and pe_h == PEY and pe_w == PEX:
                        rt_h, rt_w, pe_h, pe_w = 0, 0, 0, 0
                # else:
                #     if len(self.layer_used_pe[nlayer]) < self.hw_config.total_pe_num:
                #         self.layer_used_pe[nlayer].remove((rt_h, rt_w, pe_h, pe_w))

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
