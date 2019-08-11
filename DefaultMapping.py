from MappingMetaData import MappingMetaData
from CrossbarGridMetaData import CrossbarGridMetaData
import sys
from math import ceil
import numpy as np

class DefaultMapping(object):
    def __init__(self, hd_info, model_info):
        self.model_info = model_info
        self.hd_info = hd_info

        self.layer_list = model_info.layer_list 
        self.layer_length = len(self.layer_list) 
        self.input_h = [model_info.input_h] 
        self.input_w = [model_info.input_w] 
        self.input_c = [model_info.input_c] 
        self.filter_n = [] 
        self.filter_h = []
        self.filter_w = []
        self.filter_c = []
        self.filter_length = []
        self.pooling_h = []
        self.pooling_w = []

        for nlayer in range(self.layer_length):
            if self.layer_list[nlayer].layer_type == "convolution":
                self.filter_n.append(self.layer_list[nlayer].filter_n)
                self.filter_h.append(self.layer_list[nlayer].filter_h)
                self.filter_w.append(self.layer_list[nlayer].filter_w)
                self.filter_c.append(self.layer_list[nlayer].filter_c)
                self.filter_length.append(self.layer_list[nlayer].filter_h * self.layer_list[nlayer].filter_w * self.layer_list[nlayer].filter_c)
                self.pooling_h.append(0)
                self.pooling_w.append(0)
                self.input_h.append(self.input_h[nlayer] - self.layer_list[nlayer].filter_h + 1) # stride = 1
                self.input_w.append(self.input_w[nlayer] - self.layer_list[nlayer].filter_w + 1) # stride = 1
                self.input_c.append(self.layer_list[nlayer].filter_n)
            elif self.layer_list[nlayer].layer_type == "pooling":
                self.filter_n.append(0)
                self.filter_h.append(0)
                self.filter_w.append(0)
                self.filter_c.append(0)
                self.filter_length.append(0)
                self.pooling_h.append(self.layer_list[nlayer].pooling_h)
                self.pooling_w.append(self.layer_list[nlayer].pooling_w)
                self.input_h.append(self.input_h[nlayer] // self.layer_list[nlayer].pooling_h)
                self.input_w.append(self.input_w[nlayer] // self.layer_list[nlayer].pooling_w)
                self.input_c.append(self.input_c[nlayer])
            elif self.layer_list[nlayer].layer_type == "fully":
                self.filter_n.append(self.layer_list[nlayer].neuron_n)
                self.filter_h.append(0)
                self.filter_w.append(0)
                self.filter_c.append(0)
                self.filter_length.append(self.input_h[nlayer] * self.input_w[nlayer] * self.input_c[nlayer])
                self.pooling_h.append(0)
                self.pooling_w.append(0)
                self.input_h.append(self.layer_list[nlayer].neuron_n)
                self.input_w.append(1)
                self.input_c.append(1)

        self.crossbar_array = []   
        self.layer_mapping_to_xbar = [] 
        self.layer_mapping_to_pe = []  

        for rty_idx in range(self.hd_info.Router_num_y):
            self.crossbar_array.append([])
            self.layer_mapping_to_xbar.append([])
            self.layer_mapping_to_pe.append([])
            for rtx_idx in range(self.hd_info.Router_num_x):
                self.crossbar_array[rty_idx].append([])
                self.layer_mapping_to_xbar[rty_idx].append([])
                self.layer_mapping_to_pe[rty_idx].append([])
                for pey_idx in range(self.hd_info.PE_num_y):
                    self.crossbar_array[rty_idx][rtx_idx].append([])
                    self.layer_mapping_to_xbar[rty_idx][rtx_idx].append([])
                    self.layer_mapping_to_pe[rty_idx][rtx_idx].append([])
                    for pex_idx in range(self.hd_info.PE_num_x):
                        self.crossbar_array[rty_idx][rtx_idx][pey_idx].append([])
                        self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx].append([])
                        self.layer_mapping_to_pe[rty_idx][rtx_idx][pey_idx].append([])
                        for cuy_idx in range(self.hd_info.CU_num_y):
                            self.crossbar_array[rty_idx][rtx_idx][pey_idx][pex_idx].append([])
                            self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx].append([])
                            for cux_idx in range(self.hd_info.CU_num_x):
                                self.crossbar_array[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx].append([])
                                self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx].append([])
                                for xby_idx in range(self.hd_info.Xbar_num_y):
                                    self.crossbar_array[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx].append([])
                                    self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx].append([])
                                    for xbx_idx in range(self.hd_info.Xbar_num_x):
                                        self.crossbar_array[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx].append([])
                                        self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx].append([])
                                        for h in range(self.hd_info.Xbar_h):
                                            row = [0] * self.hd_info.Xbar_w
                                            self.crossbar_array[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx][xbx_idx].append(row)

        self.pe_mapping_dict = dict() # the rt mapping order
        ctr =  0
        for ry in range(self.hd_info.Router_num_y):
            if ry % 2 == 0:
                for rx in range(self.hd_info.Router_num_x):
                    for py in range(self.hd_info.PE_num_y):
                        for px in range(self.hd_info.PE_num_x):
                            self.pe_mapping_dict[ctr] = [ry, rx, py, px]
                            ctr += 1
            else:
                for rx in range(self.hd_info.Router_num_x-1, -1, -1):
                        for py in range(self.hd_info.PE_num_y):
                            for px in range(self.hd_info.PE_num_x):
                                self.pe_mapping_dict[ctr] = [ry, rx, py, px]
                                ctr += 1
        
        self.map()

    def map(self):
        pe_idx = 0
        for nlayer in range(self.layer_length):
            if self.model_info.layer_list[nlayer].layer_type == "convolution":
                ## Prepare weights
                matrix_height = self.filter_length[nlayer]
                matrix_width = self.filter_n[nlayer] * self.model_info.filter_bit

                pe_y_num = ceil(matrix_height / (self.hd_info.Xbar_h * self.hd_info.Xbar_num_y * self.hd_info.CU_num_y))      
                pe_x_num = ceil(matrix_width / (self.hd_info.Xbar_w * self.hd_info.Xbar_num_x * self.hd_info.CU_num_x))
                pe_total_num = pe_y_num * pe_x_num # used PE number
                
                for w in range(matrix_width):
                    for h in range(matrix_height):
                        cell_h = h % self.hd_info.Xbar_h 
                        cell_w = w % self.hd_info.Xbar_w
                        xb_h = h // self.hd_info.Xbar_h % self.hd_info.Xbar_num_y
                        xb_w = w // self.hd_info.Xbar_w % self.hd_info.Xbar_num_x
                        cu_h = h // (self.hd_info.Xbar_h * self.hd_info.Xbar_num_y) % self.hd_info.CU_num_y
                        cu_w = w // (self.hd_info.Xbar_w * self.hd_info.Xbar_num_x) % self.hd_info.CU_num_x
                        pe_h = h // (self.hd_info.Xbar_h * self.hd_info.Xbar_num_y * self.hd_info.CU_num_y)
                        pe_w = w // (self.hd_info.Xbar_w * self.hd_info.Xbar_num_x * self.hd_info.CU_num_x)
                        
                        pe_num = pe_idx + pe_h * pe_x_num + pe_w
                        rt_h = self.pe_mapping_dict[pe_num][0]
                        rt_w = self.pe_mapping_dict[pe_num][1]
                        pe_h = self.pe_mapping_dict[pe_num][2]
                        pe_w = self.pe_mapping_dict[pe_num][3]
                        
                        nfilter = w // self.model_info.filter_bit
                        nbit = w % self.model_info.filter_bit
                        ngrid = h

                        self.crossbar_array[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w][cell_h][cell_w] = CrossbarGridMetaData(nlayer, ngrid, nfilter, nbit)
                
                ## Prepare input for each xbar
                o_height = self.input_h[nlayer] - self.filter_h[nlayer] + 1  # output feature map height
                o_width = self.input_w[nlayer] - self.filter_w[nlayer] + 1   # output feature map width

                inputs = []
                for oh in range(o_height): 
                    for ow in range(o_width):
                        num_input = oh * o_width + ow
                        nn = []
                        for c in range(self.model_info.layer_list[nlayer].filter_c):
                            for h in range(self.model_info.layer_list[nlayer].filter_h):
                                for w in range(self.model_info.layer_list[nlayer].filter_w):
                                    nn.append([num_input, oh+h, ow+w, c]) # input feature map position
                        inputs.append(nn)
                inputs = np.array(inputs)

                xb_y_num = ceil(matrix_height / self.hd_info.Xbar_h)
                xb_x_num = ceil(matrix_width / self.hd_info.Xbar_w)
                
                for w in range(xb_x_num):
                    for h in range(xb_y_num): # traverse all xbar
                        xb_h = h % self.hd_info.Xbar_num_y
                        xb_w = w % self.hd_info.Xbar_num_x
                        cu_h = h // self.hd_info.Xbar_num_y % self.hd_info.CU_num_y
                        cu_w = w // self.hd_info.Xbar_num_x % self.hd_info.CU_num_x
                        pe_h = h // (self.hd_info.Xbar_num_y * self.hd_info.CU_num_y)
                        pe_w = w // (self.hd_info.Xbar_num_x * self.hd_info.CU_num_x) 
                        
                        pe_num = pe_idx + pe_h * pe_x_num + pe_w
                        rt_h = self.pe_mapping_dict[pe_num][0]
                        rt_w = self.pe_mapping_dict[pe_num][1]
                        pe_h = self.pe_mapping_dict[pe_num][2]
                        pe_w = self.pe_mapping_dict[pe_num][3]
                        
                        xbar_inputs = inputs[:, h*self.hd_info.Xbar_h : (h+1)*self.hd_info.Xbar_h].tolist()

                        if (w+1) * self.hd_info.Xbar_w <= matrix_width: 
                            xbar_column = [i for i in range(self.hd_info.Xbar_w)]
                        else: # most right
                            empt = (w+1) * self.hd_info.Xbar_w - matrix_width
                            full = self.hd_info.Xbar_w - empt
                            xbar_column = [i for i in range(full)]

                        if (h+1) * self.hd_info.Xbar_h <= matrix_height:
                            xbar_row = [i for i in range(self.hd_info.Xbar_h)] 
                        else:
                            empt = (h+1) * self.hd_info.Xbar_h - matrix_height
                            full = self.hd_info.Xbar_h - empt
                            xbar_row = [i for i in range(full)]

                        for inp in xbar_inputs:
                            self.layer_mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w].append(MappingMetaData("convolution", nlayer, xbar_row, xbar_column, inp))

                pe_idx += pe_total_num
                       
            elif self.model_info.layer_list[nlayer].layer_type == "fully":
                ## Prepare weights
                matrix_height = self.filter_length[nlayer]
                matrix_width = self.filter_n[nlayer] * self.model_info.filter_bit
                
                pe_y_num = ceil(matrix_height / (self.hd_info.Xbar_h * self.hd_info.Xbar_num_y * self.hd_info.CU_num_y))      
                pe_x_num = ceil(matrix_width / (self.hd_info.Xbar_w * self.hd_info.Xbar_num_x * self.hd_info.CU_num_x))
                pe_total_num = pe_y_num * pe_x_num # used PE num

                for w in range(matrix_width):
                    for h in range(matrix_height):
                        cell_h = h % self.hd_info.Xbar_h 
                        cell_w = w % self.hd_info.Xbar_w
                        xb_h = h // self.hd_info.Xbar_h % self.hd_info.Xbar_num_y
                        xb_w = w // self.hd_info.Xbar_w % self.hd_info.Xbar_num_x
                        cu_h = h // (self.hd_info.Xbar_h * self.hd_info.Xbar_num_y) % self.hd_info.CU_num_y
                        cu_w = w // (self.hd_info.Xbar_w * self.hd_info.Xbar_num_x) % self.hd_info.CU_num_x
                        pe_h = h // (self.hd_info.Xbar_h * self.hd_info.Xbar_num_y * self.hd_info.CU_num_y)
                        pe_w = w // (self.hd_info.Xbar_w * self.hd_info.Xbar_num_x * self.hd_info.CU_num_x)                        
                        
                        pe_num = pe_idx + pe_h * pe_x_num + pe_w
                        rt_h = self.pe_mapping_dict[pe_num][0]
                        rt_w = self.pe_mapping_dict[pe_num][1]
                        pe_h = self.pe_mapping_dict[pe_num][2]
                        pe_w = self.pe_mapping_dict[pe_num][3]
                        
                        nfilter = w // self.model_info.filter_bit
                        nbit = w % self.model_info.filter_bit
                        ngrid = h

                        self.crossbar_array[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w][cell_h][cell_w] = CrossbarGridMetaData(nlayer, ngrid, nfilter, nbit)

                ## Prepare input for each xbar   
                inputs = []
                nn = []
                for h in range(self.filter_length[nlayer]):
                    nn.append([0, h, 0, 0])
                inputs.append(nn)

                inputs = np.array(inputs)

                xb_y_num = ceil(matrix_height / self.hd_info.Xbar_h)
                xb_x_num = ceil(matrix_width / self.hd_info.Xbar_w)

                for w in range(xb_x_num):
                    for h in range(xb_y_num): # traverse all xbar
                        xb_h = h % self.hd_info.Xbar_num_y
                        xb_w = w % self.hd_info.Xbar_num_x
                        cu_h = h // self.hd_info.Xbar_num_y % self.hd_info.CU_num_y
                        cu_w = w // self.hd_info.Xbar_num_x % self.hd_info.CU_num_x
                        pe_h = h // (self.hd_info.Xbar_num_y * self.hd_info.CU_num_y)
                        pe_w = w // (self.hd_info.Xbar_num_x * self.hd_info.CU_num_x) 
                        
                        pe_num = pe_idx + pe_h * pe_x_num + pe_w
                        rt_h = self.pe_mapping_dict[pe_num][0]
                        rt_w = self.pe_mapping_dict[pe_num][1]
                        pe_h = self.pe_mapping_dict[pe_num][2]
                        pe_w = self.pe_mapping_dict[pe_num][3]

                        
                        xbar_inputs = inputs[:, h*self.hd_info.Xbar_h : (h+1)*self.hd_info.Xbar_h].tolist()

                        if (w+1) * self.hd_info.Xbar_w <= matrix_width:
                            xbar_column = [i for i in range(self.hd_info.Xbar_w)]
                        else:
                            empt = (w+1) * self.hd_info.Xbar_w - matrix_width
                            full = self.hd_info.Xbar_w - empt
                            xbar_column = [i for i in range(full)]
                        
                        if (h+1) * self.hd_info.Xbar_h <= matrix_height:
                            xbar_row = [i for i in range(self.hd_info.Xbar_h)] 
                        else:
                            empt = (h+1) * self.hd_info.Xbar_h - matrix_height
                            full = self.hd_info.Xbar_h - empt
                            xbar_row = [i for i in range(full)]

                        for inp in xbar_inputs:
                            self.layer_mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w].append(MappingMetaData("fully", nlayer, xbar_row, xbar_column, inp))
                pe_idx += pe_total_num
    
            elif self.model_info.layer_list[nlayer].layer_type == "pooling":
                o_height = self.input_h[nlayer] // self.pooling_h[nlayer]
                o_width = self.input_w[nlayer] // self.pooling_w[nlayer]

                inputs = []
                for oh in range(o_height):
                    for ow in range(o_width):
                        for c in range(self.model_info.layer_list[nlayer-1].filter_n):
                            nn = []
                            for ph in range(self.model_info.layer_list[nlayer].pooling_h):
                                for pw in range(self.model_info.layer_list[nlayer].pooling_w):
                                    nn.append([oh*self.pooling_h[nlayer] + ph, ow*self.pooling_w[nlayer] + pw, c])
                            inputs.append(nn)
                inputs = np.array(inputs)

                pe_idx -= pe_total_num
                input_per_pe = len(inputs) // pe_total_num # split into multiple pe
                if input_per_pe == 0:
                    input_per_pe = 1

                for pe_n in range(pe_total_num):
                    rt_h = self.pe_mapping_dict[pe_idx + pe_n][0]
                    rt_w = self.pe_mapping_dict[pe_idx + pe_n][1]
                    pe_h = self.pe_mapping_dict[pe_idx + pe_n][2]
                    pe_w = self.pe_mapping_dict[pe_idx + pe_n][3]

                    this_input = inputs[pe_n * input_per_pe : (pe_n+1) * input_per_pe].tolist()
                    xbar_column = [-1]
                    xbar_row = [-1]
                    if this_input:
                        self.layer_mapping_to_pe[rt_h][rt_w][pe_h][pe_w].append(MappingMetaData("pooling", nlayer, xbar_row, xbar_column, this_input))
                pe_idx += pe_total_num
             
    def __str__(self):
        return str(self.__dict__)

class ParallelismMapping(object):
    def __init__(self, hd_info, model_info):
        self.model_info = model_info
        self.hd_info = hd_info

        self.layer_list = model_info.layer_list  
        self.layer_length = len(model_info.layer_list)  
        self.input_h = [model_info.input_h] 
        self.input_w = [model_info.input_w] 
        self.input_c = [model_info.input_c]
        self.filter_n = [] 
        self.filter_h = []
        self.filter_w = []
        self.filter_c = []
        self.filter_length = []
        self.pooling_h = []
        self.pooling_w = []

        for i in range(self.layer_length):
            if self.layer_list[i].layer_type == "convolution":
                self.filter_n.append(self.layer_list[i].filter_n)
                self.filter_h.append(self.layer_list[i].filter_h)
                self.filter_w.append(self.layer_list[i].filter_w)
                self.filter_c.append(self.layer_list[i].filter_c)
                self.filter_length.append(self.layer_list[i].filter_h * self.layer_list[i].filter_w * self.layer_list[i].filter_c)
                self.pooling_h.append(0)
                self.pooling_w.append(0)
                self.input_h.append(self.input_h[i] - self.layer_list[i].filter_h + 1)
                self.input_w.append(self.input_w[i] - self.layer_list[i].filter_w + 1)
                self.input_c.append(self.layer_list[i].filter_n)
            elif self.layer_list[i].layer_type == "pooling":
                self.filter_n.append(0)
                self.filter_h.append(0)
                self.filter_w.append(0)
                self.filter_c.append(0)
                self.filter_length.append(0)
                self.pooling_h.append(self.layer_list[i].pooling_h)
                self.pooling_w.append(self.layer_list[i].pooling_w)
                self.input_h.append(self.input_h[i] // self.layer_list[i].pooling_h)
                self.input_w.append(self.input_w[i] // self.layer_list[i].pooling_w)
                self.input_c.append(self.input_c[i])
            elif self.layer_list[i].layer_type == "fully":
                self.filter_n.append(self.layer_list[i].neuron_n)
                self.filter_h.append(0)
                self.filter_w.append(0)
                self.filter_c.append(0)
                self.filter_length.append(self.input_h[i] * self.input_w[i] * self.input_c[i])
                self.pooling_h.append(0)
                self.pooling_w.append(0)
                self.input_h.append(self.filter_n[i])
                self.input_w.append(1)
                self.input_c.append(1)
        
        self.crossbar_array = []
        self.layer_mapping_to_xbar = []
        self.layer_mapping_to_pe = []

        for rty_idx in range(self.hd_info.Router_num_y):
            self.crossbar_array.append([])
            self.layer_mapping_to_xbar.append([])
            self.layer_mapping_to_pe.append([])
            for rtx_idx in range(self.hd_info.Router_num_x):
                self.crossbar_array[rty_idx].append([])
                self.layer_mapping_to_xbar[rty_idx].append([])
                self.layer_mapping_to_pe[rty_idx].append([])
                for pey_idx in range(self.hd_info.PE_num_y):
                    self.crossbar_array[rty_idx][rtx_idx].append([])
                    self.layer_mapping_to_xbar[rty_idx][rtx_idx].append([])
                    self.layer_mapping_to_pe[rty_idx][rtx_idx].append([])
                    for pex_idx in range(self.hd_info.PE_num_x):
                        self.crossbar_array[rty_idx][rtx_idx][pey_idx].append([])
                        self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx].append([])
                        self.layer_mapping_to_pe[rty_idx][rtx_idx][pey_idx].append([])
                        for cuy_idx in range(self.hd_info.CU_num_y):
                            self.crossbar_array[rty_idx][rtx_idx][pey_idx][pex_idx].append([])
                            self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx].append([])
                            for cux_idx in range(self.hd_info.CU_num_x):
                                self.crossbar_array[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx].append([])
                                self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx].append([])
                                for xby_idx in range(self.hd_info.Xbar_num_y):
                                    self.crossbar_array[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx].append([])
                                    self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx].append([])
                                    for xbx_idx in range(self.hd_info.Xbar_num_x):
                                        self.crossbar_array[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx].append([])
                                        self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx].append([])
                                        for h in range(self.hd_info.Xbar_h):
                                            row = [0] * self.hd_info.Xbar_w
                                            self.crossbar_array[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx][xbx_idx].append(row)

        self.xb_mapping_dict = dict()
        ctr =  0
        for ry in range(self.hd_info.Router_num_y):
            if ry % 2 == 0:
                for rx in range(self.hd_info.Router_num_x):
                    for py in range(self.hd_info.PE_num_y):
                        for px in range(self.hd_info.PE_num_x):
                            for cy in range(self.hd_info.CU_num_y):
                                for cx in range(self.hd_info.CU_num_x): 
                                    for xy in range(self.hd_info.Xbar_num_y):
                                        for xx in range(self.hd_info.Xbar_num_x):
                                            self.xb_mapping_dict[ctr] = [ry, rx, py, px, cy, cx, xy, xx]
                                            ctr += 1
            else:
                for rx in range(self.hd_info.Router_num_x-1, -1, -1):
                    for py in range(self.hd_info.PE_num_y):
                        for px in range(self.hd_info.PE_num_x):
                            for cy in range(self.hd_info.CU_num_y):
                                for cx in range(self.hd_info.CU_num_x): 
                                    for xy in range(self.hd_info.Xbar_num_y):
                                        for xx in range(self.hd_info.Xbar_num_x):
                                            self.xb_mapping_dict[ctr] = [ry, rx, py, px, cy, cx, xy, xx]
                                            ctr += 1

        self.map()

    def map(self):
        xbar_idx = 0
        xbar_height_start_idx = 0 
        xbar_width_start_idx = 0
        pool_pos = (0,0,0,0)
         
        for nlayer in range(self.layer_length):
            if self.model_info.layer_list[nlayer].layer_type == "convolution":
                ## PE for pooling 
                pool_pos = self.xb_mapping_dict[xbar_idx][:4]

                ## Prepare inputs
                inputs = []
                o_height = self.input_h[nlayer] - self.filter_h[nlayer] + 1  # output feature map height
                o_width = self.input_w[nlayer] - self.filter_w[nlayer] + 1   # output feature map width
                for oh in range(o_height):
                    for ow in range(o_width):
                        num_input = oh * o_width + ow
                        nn = []
                        for c in range(self.model_info.layer_list[nlayer].filter_c):
                            for h in range(self.model_info.layer_list[nlayer].filter_h):
                                for w in range(self.model_info.layer_list[nlayer].filter_w):
                                    nn.append([num_input, oh+h, ow+w, c]) # input feature map position
                        inputs.append(nn)
                inputs = np.array(inputs)

                matrix_height = self.filter_length[nlayer]
                matrix_width = self.filter_n[nlayer] * self.model_info.filter_bit
                OU_num_y = ceil(matrix_height / self.hd_info.OU_h)
                OU_num_x = ceil(matrix_width / self.hd_info.OU_w)
                for ou_idx_x in range(OU_num_x):
                    for ou_idx_y in range(OU_num_y):
                        ## OU block
                        pos = self.xb_mapping_dict[xbar_idx] # xbar mapping position
                        rt_h, rt_w = pos[0], pos[1]
                        pe_h, pe_w = pos[2], pos[3]
                        cu_h, cu_w = pos[4], pos[5]
                        xb_h, xb_w = pos[6], pos[7]

                        # OU block size
                        if ou_idx_y == OU_num_y - 1:
                            block_h = matrix_height % self.hd_info.OU_h
                            if block_h == 0:
                                block_h = self.hd_info.OU_h
                        else:
                            block_h = self.hd_info.OU_h
                        if ou_idx_x == OU_num_x - 1: 
                            block_w = matrix_width % self.hd_info.OU_w
                            if block_w == 0:
                                block_w = self.hd_info.OU_w
                        else:
                            block_w = self.hd_info.OU_w
                        
                        ## map weight
                        for b_h in range(block_h):            
                            for b_w in range(block_w):
                                w = b_w + ou_idx_x * self.hd_info.OU_w
                                h = b_h + ou_idx_y * self.hd_info.OU_h

                                nfilter = w // self.model_info.filter_bit
                                nbit = w % self.model_info.filter_bit
                                ngrid = h

                                cell_h = xbar_height_start_idx + b_h
                                cell_w = xbar_width_start_idx + b_w

                                self.crossbar_array[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w][cell_h][cell_w] = CrossbarGridMetaData(nlayer, ngrid, nfilter, nbit)
                            
                        ## Input
                        start = ou_idx_y * self.hd_info.OU_h
                        xbar_inputs = inputs[:, start:(start+block_h)].tolist()
                        xbar_column = [i for i in range(xbar_width_start_idx, xbar_width_start_idx+block_w)]
                        xbar_row = [i for i in range(xbar_height_start_idx, xbar_height_start_idx+block_h)]

                        for inp in xbar_inputs:
                            self.layer_mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w].append(MappingMetaData("convolution", nlayer, xbar_row, xbar_column, inp))
        
                        xbar_idx += 1
                        if xbar_idx >= len(self.xb_mapping_dict): 
                            xbar_idx = 0
                            xbar_height_start_idx += self.hd_info.OU_h
                            if xbar_height_start_idx + self.hd_info.OU_h > self.hd_info.Xbar_h:
                                xbar_height_start_idx = 0
                                xbar_width_start_idx += self.hd_info.OU_w

            elif self.model_info.layer_list[nlayer].layer_type == "fully":
                ## Prepare inputs
                inputs = []
                nn = []
                for h in range(self.filter_length[nlayer]):
                    nn.append([0, h, 0, 0])
                inputs.append(nn)
                inputs = np.array(inputs)
               
                matrix_height = self.filter_length[nlayer]
                matrix_width = self.filter_n[nlayer] * self.model_info.filter_bit
                OU_num_y = ceil(matrix_height / self.hd_info.OU_h)
                OU_num_x = ceil(matrix_width / self.hd_info.OU_w)
                for ou_idx_x in range(OU_num_x):
                    for ou_idx_y in range(OU_num_y):
                        ## OU block
                        # xbar mapping position
                        pos = self.xb_mapping_dict[xbar_idx]
                        rt_h, rt_w = pos[0], pos[1]
                        pe_h, pe_w = pos[2], pos[3]
                        cu_h, cu_w = pos[4], pos[5]
                        xb_h, xb_w = pos[6], pos[7]

                        # OU block size
                        if ou_idx_y == OU_num_y - 1:
                            block_h = matrix_height % self.hd_info.OU_h
                            if block_h == 0:
                                block_h = self.hd_info.OU_h
                        else:
                            block_h = self.hd_info.OU_h
                        if ou_idx_x == OU_num_x - 1: 
                            block_w = matrix_width % self.hd_info.OU_w
                            if block_w == 0:
                                block_w = self.hd_info.OU_w
                        else:
                            block_w = self.hd_info.OU_w

                        ## map weight
                        for b_h in range(block_h):            
                            for b_w in range(block_w):
                                w = b_w + ou_idx_x * self.hd_info.OU_w
                                h = b_h + ou_idx_y * self.hd_info.OU_h

                                nfilter = w // self.model_info.filter_bit
                                nbit = w % self.model_info.filter_bit
                                ngrid = h

                                cell_h = xbar_height_start_idx + b_h
                                cell_w = xbar_width_start_idx + b_w

                                self.crossbar_array[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w][cell_h][cell_w] = CrossbarGridMetaData(nlayer, ngrid, nfilter, nbit)
                            
                        ## Input
                        start = ou_idx_y * self.hd_info.OU_h
                        xbar_inputs = inputs[:, start:(start+block_h)].tolist()
                        xbar_column = [i for i in range(xbar_width_start_idx, xbar_width_start_idx+block_w)]
                        xbar_row = [i for i in range(xbar_height_start_idx, xbar_height_start_idx+block_h)]

                        for inp in xbar_inputs:
                            self.layer_mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w].append(MappingMetaData("fully", nlayer, xbar_row, xbar_column, inp))
        
                        xbar_idx += 1
                        if xbar_idx >= len(self.xb_mapping_dict): 
                            xbar_idx = 0
                            xbar_height_start_idx += self.hd_info.OU_h
                            if xbar_height_start_idx + self.hd_info.OU_h > self.hd_info.Xbar_h:
                                xbar_height_start_idx = 0
                                xbar_width_start_idx += self.hd_info.OU_w
            
            elif self.model_info.layer_list[nlayer].layer_type == "pooling":
                o_height = self.input_h[nlayer] // self.pooling_h[nlayer]
                o_width = self.input_w[nlayer] // self.pooling_w[nlayer]

                inputs = []
                for oh in range(o_height):
                    for ow in range(o_width):
                        for c in range(self.model_info.layer_list[nlayer-1].filter_n):
                            nn = []
                            for ph in range(self.model_info.layer_list[nlayer].pooling_h):
                                for pw in range(self.model_info.layer_list[nlayer].pooling_w):
                                    nn.append([oh*self.pooling_h[nlayer] + ph, ow*self.pooling_w[nlayer] + pw, c])
                            inputs.append(nn)

                rty, rtx = pool_pos[0], pool_pos[1]
                pey, pex = pool_pos[2], pool_pos[3]
                xbar_column = [-1]
                xbar_row = [-1]
                self.layer_mapping_to_pe[rty][rtx][pey][pex].append(MappingMetaData("pooling", nlayer, xbar_row, xbar_column, inputs))

    def __str__(self):
        return str(self.__dict__)

class TransferMapping(object):
    def __init__(self, hd_info, model_info):
        self.model_info = model_info
        self.hd_info = hd_info

        self.layer_list = model_info.layer_list  
        self.layer_length = len(model_info.layer_list)  
        self.input_h = [model_info.input_h]
        self.input_w = [model_info.input_w] 
        self.input_c = [model_info.input_c]
        self.filter_n = [] 
        self.filter_h = []
        self.filter_w = []
        self.filter_c = []
        self.filter_length = [] # Straighten kernel length
        self.pooling_h = []
        self.pooling_w = []

        for i in range(self.layer_length):
            if self.layer_list[i].layer_type == "convolution":
                self.filter_n.append(self.layer_list[i].filter_n)
                self.filter_h.append(self.layer_list[i].filter_h)
                self.filter_w.append(self.layer_list[i].filter_w)
                self.filter_c.append(self.layer_list[i].filter_c)
                self.filter_length.append(self.layer_list[i].filter_h * self.layer_list[i].filter_w * self.layer_list[i].filter_c)
                self.pooling_h.append(0)
                self.pooling_w.append(0)
                self.input_h.append(self.input_h[i] - self.layer_list[i].filter_h + 1)
                self.input_w.append(self.input_w[i] - self.layer_list[i].filter_w + 1)
                self.input_c.append(self.layer_list[i].filter_n)
            elif self.layer_list[i].layer_type == "pooling":
                self.filter_n.append(0)
                self.filter_h.append(0)
                self.filter_w.append(0)
                self.filter_c.append(0)
                self.filter_length.append(0)
                self.pooling_h.append(self.layer_list[i].pooling_h)
                self.pooling_w.append(self.layer_list[i].pooling_w)
                self.input_h.append(self.input_h[i] // self.layer_list[i].pooling_h)
                self.input_w.append(self.input_w[i] // self.layer_list[i].pooling_w)
                self.input_c.append(self.input_c[i])
            elif self.layer_list[i].layer_type == "fully":
                self.filter_n.append(self.layer_list[i].neuron_n)
                self.filter_h.append(0)
                self.filter_w.append(0)
                self.filter_c.append(0)
                self.filter_length.append(self.input_h[i] * self.input_w[i] * self.input_c[i])
                self.pooling_h.append(0)
                self.pooling_w.append(0)
                self.input_h.append(self.filter_n[i])
                self.input_w.append(1)
                self.input_c.append(1)

        self.crossbar_array = []
        self.layer_mapping_to_xbar = []
        self.layer_mapping_to_pe = []

        for rty_idx in range(self.hd_info.Router_num_y):
            self.crossbar_array.append([])
            self.layer_mapping_to_xbar.append([])
            self.layer_mapping_to_pe.append([])
            for rtx_idx in range(self.hd_info.Router_num_x):
                self.crossbar_array[rty_idx].append([])
                self.layer_mapping_to_xbar[rty_idx].append([])
                self.layer_mapping_to_pe[rty_idx].append([])
                for pey_idx in range(self.hd_info.PE_num_y):
                    self.crossbar_array[rty_idx][rtx_idx].append([])
                    self.layer_mapping_to_xbar[rty_idx][rtx_idx].append([])
                    self.layer_mapping_to_pe[rty_idx][rtx_idx].append([])
                    for pex_idx in range(self.hd_info.PE_num_x):
                        self.crossbar_array[rty_idx][rtx_idx][pey_idx].append([])
                        self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx].append([])
                        self.layer_mapping_to_pe[rty_idx][rtx_idx][pey_idx].append([])
                        for cuy_idx in range(self.hd_info.CU_num_y):
                            self.crossbar_array[rty_idx][rtx_idx][pey_idx][pex_idx].append([])
                            self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx].append([])
                            for cux_idx in range(self.hd_info.CU_num_x):
                                self.crossbar_array[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx].append([])
                                self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx].append([])
                                for xby_idx in range(self.hd_info.Xbar_num_y):
                                    self.crossbar_array[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx].append([])
                                    self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx].append([])
                                    for xbx_idx in range(self.hd_info.Xbar_num_x):
                                        self.crossbar_array[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx].append([])
                                        self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx].append([])
                                        for h in range(self.hd_info.Xbar_h):
                                            row = [0] * self.hd_info.Xbar_w
                                            self.crossbar_array[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx][xbx_idx].append(row)

        self.xb_mapping_dict = dict()
        self.num_of_xb_in_pe = self.hd_info.Xbar_num * self.hd_info.CU_num

        ctr =  0
        for ry in range(self.hd_info.Router_num_y):
            if ry % 2 == 0:
                for rx in range(self.hd_info.Router_num_x):
                    for py in range(self.hd_info.PE_num_y):
                        for px in range(self.hd_info.PE_num_x):
                            for cy in range(self.hd_info.CU_num_y):
                                for cx in range(self.hd_info.CU_num_x): 
                                    for xy in range(self.hd_info.Xbar_num_y):
                                        for xx in range(self.hd_info.Xbar_num_x):
                                            self.xb_mapping_dict[ctr] = [ry, rx, py, px, cy, cx, xy, xx]
                                            ctr += 1
            else:
                for rx in range(self.hd_info.Router_num_x-1, -1, -1):
                        for py in range(self.hd_info.PE_num_y):
                            for px in range(self.hd_info.PE_num_x):
                                for cy in range(self.hd_info.CU_num_y):
                                    for cx in range(self.hd_info.CU_num_x): 
                                        for xy in range(self.hd_info.Xbar_num_y):
                                            for xx in range(self.hd_info.Xbar_num_x):
                                                self.xb_mapping_dict[ctr] = [ry, rx, py, px, cy, cx, xy, xx]

        self.map()

    def map(self):
        xbar_mapping_idx = 0
        for nlayer in range(self.layer_length):
            if self.model_info.layer_list[nlayer].layer_type == "convolution":
                inputs = []
                o_height = self.input_h[nlayer] - self.filter_h[nlayer] + 1  # output feature map height
                o_width = self.input_w[nlayer] - self.filter_w[nlayer] + 1   # output feature map width
                for oh in range(o_height):
                    for ow in range(o_width):
                        num_input = oh * o_width + ow
                        nn = []
                        for c in range(self.model_info.layer_list[nlayer].filter_c):
                            for h in range(self.model_info.layer_list[nlayer].filter_h):
                                for w in range(self.model_info.layer_list[nlayer].filter_w):
                                    nn.append([num_input, oh+h, ow+w, c]) # input feature map position
                        inputs.append(nn)
                inputs = np.array(inputs)

                ## Prepare weights
                matrix_height = self.filter_length[nlayer]
                matrix_width = self.filter_n[nlayer] * self.model_info.filter_bit
                mapping_height_num_xb_per_pe = ceil(matrix_height / self.hd_info.Xbar_h)
                if mapping_height_num_xb_per_pe > self.num_of_xb_in_pe:
                    print("Mapping error: mapping_height_num_xb_per_pe > num_of_xb_in_pe.")
                    exit()
                mapping_width_num_xb_per_pe = self.num_of_xb_in_pe // mapping_height_num_xb_per_pe
                if mapping_width_num_xb_per_pe * self.hd_info.Xbar_w > matrix_width:
                    mapping_width_num_xb_per_pe = ceil(matrix_width / self.hd_info.Xbar_w)
                
                num_filter_per_pe = mapping_width_num_xb_per_pe * self.hd_info.Xbar_w // self.model_info.filter_bit
                if num_filter_per_pe > self.filter_n[nlayer]:
                    num_filter_per_pe = self.filter_n[nlayer]
                this_layer_xb_mapping_idx = xbar_mapping_idx

                for pe_n in range(ceil(self.filter_n[nlayer] / num_filter_per_pe)):
                    for xb_w_idx in range(mapping_width_num_xb_per_pe):
                        for xb_h_idx in range(mapping_height_num_xb_per_pe):
                            # XB
                            pos = self.xb_mapping_dict[this_layer_xb_mapping_idx]
                            rt_h, rt_w = pos[0], pos[1]
                            pe_h, pe_w = pos[2], pos[3]
                            cu_h, cu_w = pos[4], pos[5]
                            xb_h, xb_w = pos[6], pos[7]
                        
                            if xb_h_idx == mapping_height_num_xb_per_pe - 1:
                                height = matrix_height % self.hd_info.Xbar_h
                                if height == 0:
                                    height = self.hd_info.Xbar_h
                            else:
                                height = self.hd_info.Xbar_h

                            if xb_w_idx == mapping_width_num_xb_per_pe - 1:
                                width = (num_filter_per_pe * self.model_info.filter_bit) % self.hd_info.Xbar_w
                                if width == 0:
                                    width = self.hd_info.Xbar_w
                            else:
                                width = self.hd_info.Xbar_w
                        
                            for h in range(height):
                                for w in range(width):
                                    matrix_w = w + xb_w_idx * self.hd_info.Xbar_w + pe_n * num_filter_per_pe * self.model_info.filter_bit 
                                    matrix_h = h + xb_h_idx * self.hd_info.Xbar_h

                                    nfilter = matrix_w // self.model_info.filter_bit
                                    nbit = matrix_w % self.model_info.filter_bit
                                    ngrid = matrix_h

                                    cell_h = h
                                    cell_w = w
                                    self.crossbar_array[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w][cell_h][cell_w] = CrossbarGridMetaData(nlayer, ngrid, nfilter, nbit)
                                
                            ## Prepare inputs
                            xbar_inputs = inputs[:, xb_h_idx * self.hd_info.Xbar_h : xb_h_idx * self.hd_info.Xbar_h + height].tolist()
                            xbar_column = [i for i in range(width)]
                            xbar_row = [i for i in range(height)]
                            for inp in xbar_inputs:
                                self.layer_mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w].append(MappingMetaData("convolution", nlayer, xbar_row, xbar_column, inp))

                            this_layer_xb_mapping_idx += 1
                
                # next layer change pe
                used_pe_num = ceil(self.filter_n[nlayer] / num_filter_per_pe)
                used_xbar_num = used_pe_num * self.num_of_xb_in_pe
                xbar_mapping_idx += used_xbar_num
          
            elif self.model_info.layer_list[nlayer].layer_type == "fully":
                inputs = []
                nn = []
                for h in range(self.filter_length[nlayer]):
                    nn.append([0, h, 0, 0])
                inputs.append(nn)
                inputs = np.array(inputs)

                ## Prepare weights
                matrix_height = self.filter_length[nlayer]
                matrix_width = self.filter_n[nlayer] * self.model_info.filter_bit
                
                mapping_height_num_xb_per_pe = ceil(matrix_height / self.hd_info.Xbar_h)

                if self.num_of_xb_in_pe // mapping_height_num_xb_per_pe > matrix_width // self.hd_info.Xbar_w:
                    mapping_width_num_xb_per_pe = matrix_width // self.hd_info.Xbar_w
                else:
                    mapping_width_num_xb_per_pe = self.num_of_xb_in_pe // mapping_height_num_xb_per_pe

                num_filter_per_pe = mapping_width_num_xb_per_pe * self.hd_info.Xbar_w // self.model_info.filter_bit

                this_layer_xb_mapping_idx = xbar_mapping_idx

                for pe_n in range(ceil(self.filter_n[nlayer] / num_filter_per_pe)):
                    for xb_w_idx in range(mapping_width_num_xb_per_pe):
                        for xb_h_idx in range(mapping_height_num_xb_per_pe):
                            # XB
                            pos = self.xb_mapping_dict[this_layer_xb_mapping_idx]
                            rt_h, rt_w = pos[0], pos[1]
                            pe_h, pe_w = pos[2], pos[3]
                            cu_h, cu_w = pos[4], pos[5]
                            xb_h, xb_w = pos[6], pos[7]
                        
                            if xb_h_idx == mapping_height_num_xb_per_pe - 1:
                                height = matrix_height % self.hd_info.Xbar_h
                                if height == 0:
                                    height = self.hd_info.Xbar_h
                            else:
                                height = self.hd_info.Xbar_h

                            if xb_w_idx == mapping_width_num_xb_per_pe - 1: 
                                width = (num_filter_per_pe * self.model_info.filter_bit) % self.hd_info.Xbar_w
                                if width == 0:
                                    width = self.hd_info.Xbar_w
                            else:
                                width = self.hd_info.Xbar_w


                            for h in range(height):
                                for w in range(width):
                                    matrix_w = w + xb_w_idx * self.hd_info.Xbar_w + pe_n * num_filter_per_pe * self.model_info.filter_bit
                                    matrix_h = h + xb_h_idx * self.hd_info.Xbar_h

                                    nfilter = matrix_w // self.model_info.filter_bit
                                    nbit = matrix_w % self.model_info.filter_bit
                                    ngrid = matrix_h

                                    cell_h = h
                                    cell_w = w
                                    self.crossbar_array[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w][cell_h][cell_w] = CrossbarGridMetaData(nlayer, ngrid, nfilter, nbit)
                                
                            ## Prepare inputs
                            xbar_inputs = inputs[:, xb_h_idx * self.hd_info.Xbar_h : xb_h_idx * self.hd_info.Xbar_h + height].tolist()
                            xbar_column = [i for i in range(width)]
                            xbar_row = [i for i in range(height)]
                            for inp in xbar_inputs:
                                self.layer_mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w].append(MappingMetaData("fully", nlayer, xbar_row, xbar_column, inp))

                            this_layer_xb_mapping_idx += 1
                
                # next layer change pe
                used_pe_num = ceil(self.filter_n[nlayer] / num_filter_per_pe)
                used_xbar_num = used_pe_num * self.num_of_xb_in_pe
                xbar_mapping_idx += used_xbar_num                

            elif self.model_info.layer_list[nlayer].layer_type == "pooling":
                o_height = self.input_h[nlayer] // self.pooling_h[nlayer]
                o_width = self.input_w[nlayer] // self.pooling_w[nlayer]
                        
                inputs = []
                for oh in range(o_height):
                    for ow in range(o_width):
                        for c in range(self.model_info.layer_list[nlayer-1].filter_n):
                            nn = []
                            for ph in range(self.model_info.layer_list[nlayer].pooling_h):
                                for pw in range(self.model_info.layer_list[nlayer].pooling_w):
                                    nn.append([oh*self.pooling_h[nlayer] + ph, ow*self.pooling_w[nlayer] + pw, c])
                            inputs.append(nn)
                inputs = np.array(inputs)

                xbar_mapping_idx -= used_xbar_num
                input_per_pe = len(inputs) // used_pe_num # split into multiple pe
                if input_per_pe == 0:
                    input_per_pe = 1

                for pe_n in range(used_pe_num):
                    pos = self.xb_mapping_dict[xbar_mapping_idx + pe_n * self.num_of_xb_in_pe]
                    rt_h, rt_w = pos[0], pos[1]
                    pe_h, pe_w = pos[2], pos[3]

                    this_input = inputs[pe_n * input_per_pe : (pe_n+1) * input_per_pe].tolist()
                    xbar_column = [-1]
                    xbar_row = [-1]
                    if this_input:
                        self.layer_mapping_to_pe[rt_h][rt_w][pe_h][pe_w].append(MappingMetaData("pooling", nlayer, xbar_row, xbar_column, this_input))
                
                xbar_mapping_idx += used_xbar_num

    def __str__(self):
        return str(self.__dict__)


'''
class Paral2(object):
    def __init__(self, hd_info, model_info):
        self.model_info = model_info
        self.hd_info = hd_info

        self.layer_list = model_info.layer_list  
        self.layer_length = len(model_info.layer_list)  
        self.input_h = [model_info.input_h] 
        self.input_w = [model_info.input_w] 
        self.input_c = [model_info.input_c] 
        self.filter_n = [] 
        self.filter_h = []
        self.filter_w = []
        self.filter_c = []
        self.filter_length = [] # Straighten kernel length
        self.pooling_h = []
        self.pooling_w = []

        for i in range(self.layer_length):
            if self.layer_list[i].layer_type == "convolution":
                self.filter_n.append(self.layer_list[i].filter_n)
                self.filter_h.append(self.layer_list[i].filter_h)
                self.filter_w.append(self.layer_list[i].filter_w)
                self.filter_c.append(self.layer_list[i].filter_c)
                self.filter_length.append(self.layer_list[i].filter_h * self.layer_list[i].filter_w * self.layer_list[i].filter_c)
                self.pooling_h.append(0)
                self.pooling_w.append(0)
                self.input_h.append(self.input_h[i] - self.layer_list[i].filter_h + 1)
                self.input_w.append(self.input_w[i] - self.layer_list[i].filter_w + 1)
                self.input_c.append(self.layer_list[i].filter_n)
            elif self.layer_list[i].layer_type == "pooling":
                self.filter_n.append(0)
                self.filter_h.append(0)
                self.filter_w.append(0)
                self.filter_c.append(0)
                self.filter_length.append(0)
                self.pooling_h.append(self.layer_list[i].pooling_h)
                self.pooling_w.append(self.layer_list[i].pooling_w)
                self.input_h.append(self.input_h[i] // self.layer_list[i].pooling_h)
                self.input_w.append(self.input_w[i] // self.layer_list[i].pooling_w)
                self.input_c.append(self.input_c[i])
            elif self.layer_list[i].layer_type == "fully":
                self.filter_n.append(self.layer_list[i].neuron_n)
                self.filter_h.append(0)
                self.filter_w.append(0)
                self.filter_c.append(0)
                self.filter_length.append(self.input_h[i] * self.input_w[i] * self.input_c[i])
                self.pooling_h.append(0)
                self.pooling_w.append(0)
                self.input_h.append(self.filter_n[i])
                self.input_w.append(1)
                self.input_c.append(1)
        
        print('self.filter_n, self.filter_h, self.filter_w, self.filter_c, self.filter_length, self.input_h, self.input_w, self.input_c')
        for i in range(self.layer_length):
           print(self.layer_list[i].layer_type)
           print(self.filter_n[i], self.filter_h[i], self.filter_w[i], self.filter_c[i], self.filter_length[i], self.input_h[i], self.input_w[i], self.input_c[i])

        # initialize Xbar cell to 0
        # crossbar_array shape = (PE_idx, CU_idx, Xbar_idx, Xbar_h_idx, Xbar_w_idx)
        self.crossbar_array = []   # weights
        self.layer_mapping_to_xbar = [] # xbar's inputs
        self.layer_mapping_to_pe = []  # pooling's inputs

        for rty_idx in range(self.hd_info.Router_num_y):
            self.crossbar_array.append([])
            self.layer_mapping_to_xbar.append([])
            self.layer_mapping_to_pe.append([])
            for rtx_idx in range(self.hd_info.Router_num_x):
                self.crossbar_array[rty_idx].append([])
                self.layer_mapping_to_xbar[rty_idx].append([])
                self.layer_mapping_to_pe[rty_idx].append([])
                for pey_idx in range(self.hd_info.PE_num_y):
                    self.crossbar_array[rty_idx][rtx_idx].append([])
                    self.layer_mapping_to_xbar[rty_idx][rtx_idx].append([])
                    self.layer_mapping_to_pe[rty_idx][rtx_idx].append([])
                    for pex_idx in range(self.hd_info.PE_num_x):
                        self.crossbar_array[rty_idx][rtx_idx][pey_idx].append([])
                        self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx].append([])
                        self.layer_mapping_to_pe[rty_idx][rtx_idx][pey_idx].append([])
                        for cuy_idx in range(self.hd_info.CU_num_y):
                            self.crossbar_array[rty_idx][rtx_idx][pey_idx][pex_idx].append([])
                            self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx].append([])
                            for cux_idx in range(self.hd_info.CU_num_x):
                                self.crossbar_array[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx].append([])
                                self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx].append([])
                                for xby_idx in range(self.hd_info.Xbar_num_y):
                                    self.crossbar_array[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx].append([])
                                    self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx].append([])
                                    for xbx_idx in range(self.hd_info.Xbar_num_x):
                                        self.crossbar_array[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx].append([])
                                        self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx].append([])
                                        for h in range(self.hd_info.Xbar_h):
                                            row = [0] * self.hd_info.Xbar_w
                                            self.crossbar_array[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx][xbx_idx].append(row)

        self.xb_mapping_dict = dict()

        ctr =  0
        for ry in range(self.hd_info.Router_num_y):
            if ry % 2 == 0:
                for rx in range(self.hd_info.Router_num_x):
                    for py in range(self.hd_info.PE_num_y):
                        for px in range(self.hd_info.PE_num_x):
                            for cy in range(self.hd_info.CU_num_y):
                                for cx in range(self.hd_info.CU_num_x): 
                                    for xy in range(self.hd_info.Xbar_num_y):
                                        for xx in range(self.hd_info.Xbar_num_x):
                                            self.xb_mapping_dict[ctr] = [ry, rx, py, px, cy, cx, xy, xx]
                                            ctr += 1
            else:
                for rx in range(self.hd_info.Router_num_x-1, -1, -1):
                    for py in range(self.hd_info.PE_num_y):
                        for px in range(self.hd_info.PE_num_x):
                            for cy in range(self.hd_info.CU_num_y):
                                for cx in range(self.hd_info.CU_num_x): 
                                    for xy in range(self.hd_info.Xbar_num_y):
                                        for xx in range(self.hd_info.Xbar_num_x):
                                            self.xb_mapping_dict[ctr] = [ry, rx, py, px, cy, cx, xy, xx]
                                            ctr += 1
        print(self.xb_mapping_dict)

        self.map()
        print(self.crossbar_array[0][0][0][0][0][0][0][0][0][2])

    def map(self):
        xbar_idx = 0 # OU_col block在哪一個XBAR開始放
        xbar_col_idx = 0 # OU_col block要從哪個column開始放
        for nlayer in range(self.layer_length):
            if self.model_info.layer_list[nlayer].layer_type == "convolution":
                # Prepare weights
                matrix_height = self.filter_length[nlayer]
                matrix_width = self.filter_n[nlayer] * self.model_info.filter_bit
                print(matrix_height, matrix_width)
                
                num_of_XB_per_ou_col_block = ceil(matrix_height/self.hd_info.Xbar_h)
                print(num_of_XB_per_ou_col_block)
                
                OU_col_num = ceil(matrix_width / self.hd_info.OU_w)
                print(OU_col_num)

                # Prepaper inputs
                inputs = []
                o_height = self.input_h[nlayer] - self.filter_h[nlayer] + 1  # output feature map height
                o_width = self.input_w[nlayer] - self.filter_w[nlayer] + 1   # output feature map width
                for oh in range(o_height):
                    for ow in range(o_width):
                        num_input = oh * o_width + ow
                        nn = []
                        for c in range(self.model_info.layer_list[nlayer].filter_c):
                            for h in range(self.model_info.layer_list[nlayer].filter_h):
                                for w in range(self.model_info.layer_list[nlayer].filter_w):
                                    nn.append([num_input, oh+h, ow+w, c]) # input feature map position
                        inputs.append(nn)
                inputs = np.array(inputs)
                print(inputs)

                for OU_col_idx in range(OU_col_num):
                    這裡就是一個OU_col的block了
                    for h in range(matrix_height):
                        xb_shift = h // self.hd_info.Xbar_h
                        
                        if OU_col_idx == OU_col_num-1:
                            OW = matrix_width % self.hd_info.OU_w
                            if OW == 0:
                                OW = self.hd_info.OU_w
                        else:
                            OW = self.hd_info.OU_w
                        for ow in range(OW):
                            一個cell
                            w = ow + OU_col_idx * self.hd_info.OU_w
                            nfilter = w // self.model_info.filter_bit
                            nbit = w % self.model_info.filter_bit
                            ngrid = h

                            pos = self.xb_mapping_dict[xbar_idx + xb_shift]
                            rt_h, rt_w = pos[0], pos[1]
                            pe_h, pe_w = pos[2], pos[3]
                            cu_h, cu_w = pos[4], pos[5]
                            xb_h, xb_w = pos[6], pos[7]
                            cell_h = h % self.hd_info.Xbar_h
                            cell_w = xbar_col_idx + ow
                            self.crossbar_array[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w][cell_h][cell_w] = CrossbarGridMetaData(nlayer, ngrid, nfilter, nbit)
                        
                    ## Input
                    for xb_c in range(num_of_XB_per_ou_col_block): # 一個一個xbar加入input
                        pos = self.xb_mapping_dict[xbar_idx + xb_c]
                        rt_h, rt_w = pos[0], pos[1]
                        pe_h, pe_w = pos[2], pos[3]
                        cu_h, cu_w = pos[4], pos[5]
                        xb_h, xb_w = pos[6], pos[7]

                        xbar_inputs = inputs[:, xb_c*self.hd_info.Xbar_h: (xb_c+1)*self.hd_info.Xbar_h].tolist()

                        if OU_col_idx == OU_col_num-1:
                            OW = matrix_width % self.hd_info.OU_w 
                            if OW == 0:
                                OW = self.hd_info.OU_w
                        else:
                            OW = self.hd_info.OU_w
                        
                        xbar_column = [i for i in range(OW)]
                        print(xbar_column)
                        print(xbar_idx + xb_c, xbar_inputs, xbar_column)

                        self.layer_mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w].append(MappingMetaData("convolution", nlayer, xbar_column, xbar_inputs))
      
                    xbar_idx += num_of_XB_per_ou_col_block #下一個OU_col block要放的XBAR位置
                    if xbar_idx >= len(self.xb_mapping_dict): #所有的XB都過一輪了
                        xbar_idx = 0
                        xbar_col_idx += self.hd_info.OU_w
                        print(xbar_col_idx)

            elif self.model_info.layer_list[nlayer].layer_type == "pooling":
                ### Default pooling 有錯已修正 這邊沒檢查

                o_height = self.input_h[nlayer] // self.pooling_h[nlayer]
                o_width = self.input_w[nlayer] // self.pooling_w[nlayer]
                print(nlayer, o_height, o_width)
                inputs = []
                for i in range(o_height):
                    for j in range(o_width):
                        for c in range(self.model_info.layer_list[nlayer-1].filter_n):
                            nn = []
                            for k in range(self.model_info.layer_list[nlayer].pooling_h):
                                for l in range(self.model_info.layer_list[nlayer].pooling_w):
                                    nn.append([i*self.pooling_h[nlayer] + k, j*self.pooling_w[nlayer] + l, c])
                            inputs.append(nn)
                inputs = np.array(inputs)
                print(nlayer, inputs)

                xbar_column = [-1]
                self.layer_mapping_to_pe[0][0][0][0].append(MappingMetaData("pooling", nlayer, xbar_column, inputs))


            elif self.model_info.layer_list[nlayer].layer_type == "fully":
                # Prepare weights
                matrix_height = self.filter_length[nlayer]
                matrix_width = self.filter_n[nlayer] * self.model_info.filter_bit
                print(matrix_height, matrix_width)
                
                num_of_XB_per_ou_col_block = ceil(matrix_height/self.hd_info.Xbar_h)
                print(num_of_XB_per_ou_col_block)
                
                OU_col_num = ceil(matrix_width / self.hd_info.OU_w)
                print(OU_col_num)

                # Prepaper inputs
                inputs = []
                nn = []
                for h in range(self.filter_length[nlayer]):
                    nn.append([0, h, 0, 0])
                inputs.append(nn)
                print(inputs)

                inputs = np.array(inputs)
                
                print(OU_col_num)
                for OU_col_idx in range(OU_col_num):
                    這裡就是一個OU_col的block了
                    for h in range(matrix_height):
                        xb_shift = h // self.hd_info.Xbar_h
                        print(xb_shift)
                        
                        if OU_col_idx == OU_col_num-1:
                            OW = matrix_width % self.hd_info.OU_w
                            if OW == 0:
                                OW = self.hd_info.OU_w
                        else:
                            OW = self.hd_info.OU_w
                        for ow in range(OW):
                            一個cell

                            w = ow + OU_col_idx * self.hd_info.OU_w
                            nfilter = w // self.model_info.filter_bit
                            nbit = w % self.model_info.filter_bit
                            ngrid = h
                            print(matrix_w, nfilter, nbit, ngrid)

                            pos = self.xb_mapping_dict[xbar_idx + xb_shift]
                            rt_h, rt_w = pos[0], pos[1]
                            pe_h, pe_w = pos[2], pos[3]
                            cu_h, cu_w = pos[4], pos[5]
                            xb_h, xb_w = pos[6], pos[7]
                            cell_h = h % self.hd_info.Xbar_h
                            cell_w = xbar_col_idx + ow

                            self.crossbar_array[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w][cell_h][cell_w] = CrossbarGridMetaData(nlayer, ngrid, nfilter, nbit)
                    
                    ## Input 
                    for xb_c in range(num_of_XB_per_ou_col_block): # 一個一個xbar加入input
                        pos = self.xb_mapping_dict[xbar_idx + xb_c]
                        rt_h, rt_w = pos[0], pos[1]
                        pe_h, pe_w = pos[2], pos[3]
                        cu_h, cu_w = pos[4], pos[5]
                        xb_h, xb_w = pos[6], pos[7]

                        xbar_inputs = inputs[:, xb_c*self.hd_info.Xbar_h: (xb_c+1)*self.hd_info.Xbar_h].tolist()
                        
                        if OU_col_idx == OU_col_num-1:
                            OW = matrix_width % self.hd_info.OU_w
                            if OW == 0:
                                OW = self.hd_info.OU_w
                        else:
                            OW = self.hd_info.OU_w
                        xbar_column = [i for i in range(OW)] 
                        print(xbar_column)
                        print(xbar_idx + xb_c, xbar_inputs, xbar_column)

                        self.layer_mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w].append(MappingMetaData("fully", nlayer, xbar_column, xbar_inputs))
      
                    xbar_idx += num_of_XB_per_ou_col_block #下一個OU_col block要放的XBAR位置
                    if xbar_idx >= len(self.xb_mapping_dict): #所有的XB都過一輪了
                        xbar_idx = 0
                        xbar_col_idx += self.hd_info.OU_w


    def __str__(self):
        return str(self.__dict__)

'''
