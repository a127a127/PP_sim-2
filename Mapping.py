from HardwareMetaData import HardwareMetaData
from configs.ModelConfig import ModelConfig
from Model import Model
from MappingMetaData import MappingMetaData
from CrossbarGridMetaData import CrossbarGridMetaData
import sys
from math import ceil
import numpy as np
import gc

class DefaultMapping(object):
    def __init__(self):
        model_config = ModelConfig()
        self.model_info =  Model(model_config)
        self.hd_info = HardwareMetaData()

        self.crossbar_array = np.zeros((self.hd_info.Router_num_y, self.hd_info.Router_num_x, 
                                        self.hd_info.PE_num_y, self.hd_info.PE_num_x,
                                        self.hd_info.CU_num_y, self.hd_info.CU_num_x,
                                        self.hd_info.Xbar_num_y, self.hd_info.Xbar_num_x, 
                                        self.hd_info.Xbar_h, self.hd_info.Xbar_w, 4), dtype=np.int)
        self.layer_mapping_to_xbar = [] 
        self.layer_mapping_to_pe = []
        for rty_idx in range(self.hd_info.Router_num_y):
            self.layer_mapping_to_xbar.append([])
            self.layer_mapping_to_pe.append([])
            for rtx_idx in range(self.hd_info.Router_num_x):
                self.layer_mapping_to_xbar[rty_idx].append([])
                self.layer_mapping_to_pe[rty_idx].append([])
                for pey_idx in range(self.hd_info.PE_num_y):
                    self.layer_mapping_to_xbar[rty_idx][rtx_idx].append([])
                    self.layer_mapping_to_pe[rty_idx][rtx_idx].append([])
                    for pex_idx in range(self.hd_info.PE_num_x):
                        self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx].append([])
                        self.layer_mapping_to_pe[rty_idx][rtx_idx][pey_idx].append([])
                        for cuy_idx in range(self.hd_info.CU_num_y):
                            self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx].append([])
                            for cux_idx in range(self.hd_info.CU_num_x):
                                self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx].append([])
                                for xby_idx in range(self.hd_info.Xbar_num_y):
                                    self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx].append([])
                                    for xbx_idx in range(self.hd_info.Xbar_num_x):
                                        self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx].append([])

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
        pe_total_num = 1
        for nlayer in range(self.model_info.layer_length):
            if self.model_info.layer_list[nlayer].layer_type == "convolution":
                print("Convolution", nlayer)
                ## Weights
                matrix_height = self.model_info.filter_length[nlayer]
                cells_per_filter = ceil(self.model_info.filter_bit / self.hd_info.cell_bit_width)
                matrix_width = cells_per_filter * self.model_info.filter_n[nlayer]

                pe_y_num = ceil(matrix_height / (self.hd_info.Xbar_h * self.hd_info.Xbar_num_y * self.hd_info.CU_num_y))      
                pe_x_num = ceil(matrix_width / (self.hd_info.Xbar_w * self.hd_info.Xbar_num_x * self.hd_info.CU_num_x))
                pe_total_num = pe_y_num * pe_x_num # used PE number

                for w in range(matrix_width):
                    for h in range(matrix_height):
                        # traverse all cells
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

                        nfilter = w // cells_per_filter
                        start_bit = w % cells_per_filter * self.hd_info.cell_bit_width
                        # end_bit = start_bit + self.hd_info.cell_bit_width
                        # if end_bit >= self.model_info.filter_bit:
                        #     end_bit = self.model_info.filter_bit
                        # nbit = [i for i in range(start_bit, end_bit)]
                        nbit = start_bit
                        ngrid = h

                        #self.crossbar_array[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w][cell_h][cell_w] = CrossbarGridMetaData(nlayer, ngrid, nfilter, nbit)
                        self.crossbar_array[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w][cell_h][cell_w] = [nlayer, ngrid, nfilter, nbit]

                ## Inputs
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
                                    nn.append([num_input, oh*strides+h, ow*strides+w, c]) # input feature map position (padding後的)
                        inputs.append(nn)
                inputs = np.array(inputs)

                xb_y_num = ceil(matrix_height / self.hd_info.Xbar_h)
                xb_x_num = ceil(matrix_width / self.hd_info.Xbar_w)

                for w in range(xb_x_num):
                    for h in range(xb_y_num):
                        # traverse all xbar
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
                            self.layer_mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w].append(MappingMetaData("convolution", nlayer, xbar_row, xbar_column, inp))

                pe_idx += pe_total_num

            elif self.model_info.layer_list[nlayer].layer_type == "fully":
                print("Fully", nlayer)
                ## Weights
                matrix_height = self.model_info.filter_length[nlayer]
                cells_per_filter = ceil(self.model_info.filter_bit / self.hd_info.cell_bit_width)
                matrix_width = cells_per_filter * self.model_info.filter_n[nlayer]

                pe_y_num = ceil(matrix_height / (self.hd_info.Xbar_h * self.hd_info.Xbar_num_y * self.hd_info.CU_num_y))      
                pe_x_num = ceil(matrix_width / (self.hd_info.Xbar_w * self.hd_info.Xbar_num_x * self.hd_info.CU_num_x))
                pe_total_num = pe_y_num * pe_x_num # used PE num

                for w in range(matrix_width):
                    for h in range(matrix_height):
                        # traverse all cells
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

                        nfilter = w // cells_per_filter
                        start_bit = w % cells_per_filter * self.hd_info.cell_bit_width
                        # end_bit = start_bit + self.hd_info.cell_bit_width
                        # if end_bit >= self.model_info.filter_bit:
                        #     end_bit = self.model_info.filter_bit
                        # nbit = [i for i in range(start_bit, end_bit)]
                        nbit = start_bit
                        ngrid = h

                        #self.crossbar_array[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w][cell_h][cell_w] = CrossbarGridMetaData(nlayer, ngrid, nfilter, nbit)
                        self.crossbar_array[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w][cell_h][cell_w] = [nlayer, ngrid, nfilter, nbit]

                ## Inputs   
                inputs = []
                nn = []
                for h in range(self.model_info.filter_length[nlayer]):
                    nn.append([0, h, 0, 0])
                inputs.append(nn)

                inputs = np.array(inputs)

                xb_y_num = ceil(matrix_height / self.hd_info.Xbar_h)
                xb_x_num = ceil(matrix_width / self.hd_info.Xbar_w)

                for w in range(xb_x_num):
                    for h in range(xb_y_num):
                        # traverse all xbar
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
                print("Pooling", nlayer)
                o_height = self.model_info.input_h[nlayer] // self.model_info.pooling_h[nlayer]
                o_width = self.model_info.input_w[nlayer] // self.model_info.pooling_w[nlayer]
                inputs = []
                for oh in range(o_height):
                    for ow in range(o_width):
                        for c in range(self.model_info.input_c[nlayer]):
                            nn = []
                            for ph in range(self.model_info.pooling_h[nlayer]):
                                for pw in range(self.model_info.pooling_w[nlayer]):
                                    nn.append([oh * self.model_info.pooling_h[nlayer] + ph, ow * self.model_info.pooling_w[nlayer] + pw, c])
                            inputs.append(nn)
                inputs = np.array(inputs)

                pe_idx -= pe_total_num
                if pe_idx < 0:
                    pe_idx = 0
                input_per_pe = len(inputs) // pe_total_num # split into multiple pe
                if input_per_pe == 0:
                    input_per_pe = 1

                for pe_n in range(pe_total_num):
                    rt_h = self.pe_mapping_dict[pe_idx + pe_n][0]
                    rt_w = self.pe_mapping_dict[pe_idx + pe_n][1]
                    pe_h = self.pe_mapping_dict[pe_idx + pe_n][2]
                    pe_w = self.pe_mapping_dict[pe_idx + pe_n][3]

                    if pe_n + 1 == pe_total_num:
                        this_input = inputs[pe_n * input_per_pe : ].tolist()
                    else:
                        this_input = inputs[pe_n * input_per_pe : (pe_n+1) * input_per_pe].tolist()
                    xbar_column = [-1]
                    xbar_row = [-1]
                    if this_input:
                        self.layer_mapping_to_pe[rt_h][rt_w][pe_h][pe_w].append(MappingMetaData("pooling", nlayer, xbar_row, xbar_column, this_input))
                pe_idx += pe_total_num

    def __str__(self):
        return str(self.__dict__)

class HighParallelismMapping(object):
    def __init__(self):
        model_config = ModelConfig()
        self.model_info = Model(model_config)
        self.hd_info = HardwareMetaData()

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
                                            #row = [0] * self.hd_info.Xbar_w
                                            row = [[-1,-1,-1,-1] for i in range(self.hd_info.Xbar_w)]
                                            self.crossbar_array[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx][xbx_idx].append(row)
        self.crossbar_array = np.array(self.crossbar_array)

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
        pool_pos = [0, 0, 0, 0]
        for nlayer in range(self.model_info.layer_length):
            if self.model_info.layer_list[nlayer].layer_type == "convolution":
                print("Convolution", nlayer)
                ## PE for pooling 
                pool_pos = self.xb_mapping_dict[xbar_idx][:4]

                ## Inputs
                strides = self.model_info.strides[nlayer]
                inputs = []
                o_height = self.model_info.input_h[nlayer] - self.model_info.filter_h[nlayer] + 1  # output feature map height, slide = 1
                o_width = self.model_info.input_w[nlayer] - self.model_info.filter_w[nlayer] + 1   # output feature map width, slide = 1
                for oh in range(o_height):
                    for ow in range(o_width):
                        num_input = oh * o_width + ow
                        nn = []
                        for c in range(self.model_info.filter_c[nlayer]):
                            for h in range(self.model_info.filter_h[nlayer]):
                                for w in range(self.model_info.filter_w[nlayer]):
                                    nn.append([num_input, oh*strides+h, ow*strides+w, c]) # input feature map position (padding後的)
                        inputs.append(nn)
                inputs = np.array(inputs)

                matrix_height = self.model_info.filter_length[nlayer]
                cells_per_filter = ceil(self.model_info.filter_bit / self.hd_info.cell_bit_width)
                matrix_width = cells_per_filter * self.model_info.filter_n[nlayer]

                OU_num_y = ceil(matrix_height / self.hd_info.OU_h)
                OU_num_x = ceil(matrix_width / self.hd_info.OU_w)
                for ou_idx_x in range(OU_num_x):
                    for ou_idx_y in range(OU_num_y):
                        pos = self.xb_mapping_dict[xbar_idx] # xbar mapping position
                        rt_h, rt_w = pos[0], pos[1]
                        pe_h, pe_w = pos[2], pos[3]
                        cu_h, cu_w = pos[4], pos[5]
                        xb_h, xb_w = pos[6], pos[7]

                        # OU block size
                        if ou_idx_y + 1 == OU_num_y:
                            block_h = matrix_height - ou_idx_y * self.hd_info.OU_h
                        else:
                            block_h = self.hd_info.OU_h
                        if ou_idx_x + 1 == OU_num_x: 
                            block_w = matrix_width - ou_idx_x * self.hd_info.OU_w
                        else:
                            block_w = self.hd_info.OU_w
                        
                        ## Weight
                        for b_h in range(block_h):            
                            for b_w in range(block_w):
                                w = b_w + ou_idx_x * self.hd_info.OU_w
                                h = b_h + ou_idx_y * self.hd_info.OU_h

                                nfilter = w // cells_per_filter
                                start_bit = w % cells_per_filter * self.hd_info.cell_bit_width
                                # end_bit = start_bit + self.hd_info.cell_bit_width
                                # if end_bit >= self.model_info.filter_bit:
                                #     end_bit = self.model_info.filter_bit
                                # nbit = [i for i in range(start_bit, end_bit)]
                                nbit = start_bit
                                ngrid = h

                                cell_h = xbar_height_start_idx + b_h
                                cell_w = xbar_width_start_idx + b_w

                                #self.crossbar_array[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w][cell_h][cell_w] = CrossbarGridMetaData(nlayer, ngrid, nfilter, nbit)
                                self.crossbar_array[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w][cell_h][cell_w] = [nlayer, ngrid, nfilter, nbit]
                            
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
                print("Fully", nlayer)
                ## Inputs
                inputs = []
                nn = []
                for h in range(self.model_info.filter_length[nlayer]):
                    nn.append([0, h, 0, 0])
                inputs.append(nn)
                inputs = np.array(inputs)
               
                matrix_height = self.model_info.filter_length[nlayer]
                # matrix_width = self.model_info.filter_n[nlayer] * self.model_info.filter_bit
                cells_per_filter = ceil(self.model_info.filter_bit / self.hd_info.cell_bit_width)
                matrix_width = cells_per_filter * self.model_info.filter_n[nlayer]
                OU_num_y = ceil(matrix_height / self.hd_info.OU_h)
                OU_num_x = ceil(matrix_width / self.hd_info.OU_w)
                for ou_idx_x in range(OU_num_x):
                    for ou_idx_y in range(OU_num_y):
                        pos = self.xb_mapping_dict[xbar_idx]
                        rt_h, rt_w = pos[0], pos[1]
                        pe_h, pe_w = pos[2], pos[3]
                        cu_h, cu_w = pos[4], pos[5]
                        xb_h, xb_w = pos[6], pos[7]

                        # OU block size
                        if ou_idx_y + 1 == OU_num_y:
                            block_h = matrix_height - ou_idx_y * self.hd_info.OU_h
                        else:
                            block_h = self.hd_info.OU_h
                        if ou_idx_x + 1 == OU_num_x: 
                            block_w = matrix_width - ou_idx_x * self.hd_info.OU_w
                        else:
                            block_w = self.hd_info.OU_w

                        ## Weight
                        for b_h in range(block_h):            
                            for b_w in range(block_w):
                                w = b_w + ou_idx_x * self.hd_info.OU_w
                                h = b_h + ou_idx_y * self.hd_info.OU_h

                                # nfilter = w // self.model_info.filter_bit
                                # nbit = w % self.model_info.filter_bit
                                nfilter = w // cells_per_filter
                                start_bit = w % cells_per_filter * self.hd_info.cell_bit_width
                                # end_bit = start_bit + self.hd_info.cell_bit_width
                                # if end_bit >= self.model_info.filter_bit:
                                #     end_bit = self.model_info.filter_bit
                                # nbit = [i for i in range(start_bit, end_bit)]
                                nbit = start_bit
                                ngrid = h


                                cell_h = xbar_height_start_idx + b_h
                                cell_w = xbar_width_start_idx + b_w

                                #self.crossbar_array[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w][cell_h][cell_w] = CrossbarGridMetaData(nlayer, ngrid, nfilter, nbit)
                                self.crossbar_array[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w][cell_h][cell_w] = [nlayer, ngrid, nfilter, nbit]
                        
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
                print("Pooling", nlayer)
                o_height = self.model_info.input_h[nlayer] // self.model_info.pooling_h[nlayer]
                o_width = self.model_info.input_w[nlayer] // self.model_info.pooling_w[nlayer]

                inputs = []
                for oh in range(o_height):
                    for ow in range(o_width):
                        for c in range(self.model_info.filter_n[nlayer-1]):
                            nn = []
                            for ph in range(self.model_info.pooling_h[nlayer]):
                                for pw in range(self.model_info.pooling_w[nlayer]):
                                    nn.append([oh * self.model_info.pooling_h[nlayer] + ph, ow * self.model_info.pooling_w[nlayer] + pw, c])
                            inputs.append(nn)

                rty, rtx = pool_pos[0], pool_pos[1]
                pey, pex = pool_pos[2], pool_pos[3]
                xbar_column = [-1]
                xbar_row = [-1]

                self.layer_mapping_to_pe[rty][rtx][pey][pex].append(MappingMetaData("pooling", nlayer, xbar_row, xbar_column, inputs))

    def __str__(self):
        return str(self.__dict__)

class SameColumnFirstMapping(object):
    def __init__(self):
        model_config = ModelConfig()
        self.model_info = Model(model_config)
        self.hd_info = HardwareMetaData()

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
                                            #row = [0] * self.hd_info.Xbar_w
                                            row = [[-1,-1,-1,-1] for i in range(self.hd_info.Xbar_w)]
                                            self.crossbar_array[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx][xbx_idx].append(row)
        self.crossbar_array = np.array(self.crossbar_array)

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
        
        for nlayer in range(self.model_info.layer_length):
            if self.model_info.layer_list[nlayer].layer_type == "convolution":
                print("Convolution", nlayer)
                ## Inputs
                strides = self.model_info.strides[nlayer]
                inputs = []
                o_height = self.model_info.input_h[nlayer] - self.model_info.filter_h[nlayer] + 1  # output feature map height
                o_width = self.model_info.input_w[nlayer] - self.model_info.filter_w[nlayer] + 1   # output feature map width
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

                ## Weights
                matrix_height = self.model_info.filter_length[nlayer]
                # matrix_width = self.model_info.filter_n[nlayer] * self.model_info.filter_bit
                cells_per_filter = ceil(self.model_info.filter_bit / self.hd_info.cell_bit_width)
                matrix_width = cells_per_filter * self.model_info.filter_n[nlayer]

                mapping_height_num_xb_per_pe = ceil(matrix_height / self.hd_info.Xbar_h)
                if mapping_height_num_xb_per_pe > self.num_of_xb_in_pe:
                    print("Mapping error: mapping_height_num_xb_per_pe > num_of_xb_in_pe.")
                    exit()
                mapping_width_num_xb_per_pe = self.num_of_xb_in_pe // mapping_height_num_xb_per_pe
                if mapping_width_num_xb_per_pe * self.hd_info.Xbar_w > matrix_width:
                    mapping_width_num_xb_per_pe = ceil(matrix_width / self.hd_info.Xbar_w)
                
                # num_filter_per_pe = mapping_width_num_xb_per_pe * self.hd_info.Xbar_w // self.model_info.filter_bit
                num_filter_per_pe = mapping_width_num_xb_per_pe * self.hd_info.Xbar_w // cells_per_filter
                if num_filter_per_pe > self.model_info.filter_n[nlayer]:
                    num_filter_per_pe = self.model_info.filter_n[nlayer]
                this_layer_xb_mapping_idx = xbar_mapping_idx

                for pe_n in range(ceil(self.model_info.filter_n[nlayer] / num_filter_per_pe)):
                    for xb_w_idx in range(mapping_width_num_xb_per_pe):
                        for xb_h_idx in range(mapping_height_num_xb_per_pe):
                            pos = self.xb_mapping_dict[this_layer_xb_mapping_idx]
                            rt_h, rt_w = pos[0], pos[1]
                            pe_h, pe_w = pos[2], pos[3]
                            cu_h, cu_w = pos[4], pos[5]
                            xb_h, xb_w = pos[6], pos[7]

                            if xb_h_idx + 1 == mapping_height_num_xb_per_pe:
                                block_height = matrix_height - xb_h_idx * self.hd_info.Xbar_h
                            else:
                                block_height = self.hd_info.Xbar_h

                            if xb_w_idx + 1 == mapping_width_num_xb_per_pe:
                                # block_width = (num_filter_per_pe * self.model_info.filter_bit) - xb_w_idx * self.hd_info.Xbar_w
                                block_width = (num_filter_per_pe * cells_per_filter) - xb_w_idx * self.hd_info.Xbar_w
                            else:
                                block_width = self.hd_info.Xbar_w

                            for bh in range(block_height):
                                for bw in range(block_width):
                                    w = bw + xb_w_idx * self.hd_info.Xbar_w + pe_n * num_filter_per_pe * cells_per_filter
                                    h = bh + xb_h_idx * self.hd_info.Xbar_h

                                    nfilter = w // cells_per_filter
                                    start_bit = w % cells_per_filter * self.hd_info.cell_bit_width
                                    # end_bit = start_bit + self.hd_info.cell_bit_width
                                    # if end_bit >= self.model_info.filter_bit:
                                    #     end_bit = self.model_info.filter_bit
                                    # nbit = [i for i in range(start_bit, end_bit)]
                                    nbit = start_bit
                                    ngrid = h

                                    cell_h = bh
                                    cell_w = bw
                                    #self.crossbar_array[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w][cell_h][cell_w] = CrossbarGridMetaData(nlayer, ngrid, nfilter, nbit)
                                    self.crossbar_array[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w][cell_h][cell_w] = [nlayer, ngrid, nfilter, nbit]

                            ## Inputs
                            xbar_inputs = inputs[:, xb_h_idx * self.hd_info.Xbar_h : xb_h_idx * self.hd_info.Xbar_h + block_height].tolist()
                            xbar_column = [i for i in range(block_width)]
                            xbar_row = [i for i in range(block_height)]
                            for inp in xbar_inputs:
                                self.layer_mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w].append(MappingMetaData("convolution", nlayer, xbar_row, xbar_column, inp))

                            this_layer_xb_mapping_idx += 1

                # next layer change pe
                used_pe_num = ceil(self.model_info.filter_n[nlayer] / num_filter_per_pe)
                used_xbar_num = used_pe_num * self.num_of_xb_in_pe
                xbar_mapping_idx += used_xbar_num

            elif self.model_info.layer_list[nlayer].layer_type == "fully":
                print("Fully", nlayer)
                ## Inputs
                inputs = []
                nn = []
                for h in range(self.model_info.filter_length[nlayer]):
                    nn.append([0, h, 0, 0])
                inputs.append(nn)
                inputs = np.array(inputs)

                ## Weights
                matrix_height = self.model_info.filter_length[nlayer]
                # matrix_width = self.model_info.filter_n[nlayer] * self.model_info.filter_bit
                cells_per_filter = ceil(self.model_info.filter_bit / self.hd_info.cell_bit_width)
                matrix_width = cells_per_filter * self.model_info.filter_n[nlayer]

                mapping_height_num_xb_per_pe = ceil(matrix_height / self.hd_info.Xbar_h)
                if mapping_height_num_xb_per_pe > self.num_of_xb_in_pe:
                    print("Mapping error: mapping_height_num_xb_per_pe > num_of_xb_in_pe.")
                    exit()
                mapping_width_num_xb_per_pe = self.num_of_xb_in_pe // mapping_height_num_xb_per_pe
                if mapping_width_num_xb_per_pe * self.hd_info.Xbar_w > matrix_width:
                    mapping_width_num_xb_per_pe = ceil(matrix_width / self.hd_info.Xbar_w)

                # num_filter_per_pe = mapping_width_num_xb_per_pe * self.hd_info.Xbar_w // self.model_info.filter_bit
                num_filter_per_pe = mapping_width_num_xb_per_pe * self.hd_info.Xbar_w // cells_per_filter
                if num_filter_per_pe > self.model_info.filter_n[nlayer]:
                    num_filter_per_pe = self.model_info.filter_n[nlayer]
                this_layer_xb_mapping_idx = xbar_mapping_idx

                for pe_n in range(ceil(self.model_info.filter_n[nlayer] / num_filter_per_pe)):
                    for xb_w_idx in range(mapping_width_num_xb_per_pe):
                        for xb_h_idx in range(mapping_height_num_xb_per_pe):
                            pos = self.xb_mapping_dict[this_layer_xb_mapping_idx]
                            rt_h, rt_w = pos[0], pos[1]
                            pe_h, pe_w = pos[2], pos[3]
                            cu_h, cu_w = pos[4], pos[5]
                            xb_h, xb_w = pos[6], pos[7]

                            if xb_h_idx + 1 == mapping_height_num_xb_per_pe:
                                block_height = matrix_height - xb_h_idx * self.hd_info.Xbar_h
                            else:
                                block_height = self.hd_info.Xbar_h

                            if xb_w_idx + 1 == mapping_width_num_xb_per_pe: 
                                # block_width = (num_filter_per_pe * self.model_info.filter_bit) - xb_w_idx * self.hd_info.Xbar_w
                                block_width = (num_filter_per_pe * cells_per_filter) - xb_w_idx * self.hd_info.Xbar_w
                            else:
                                block_width = self.hd_info.Xbar_w

                            for bh in range(block_height):
                                for bw in range(block_width):
                                    # w = bw + xb_w_idx * self.hd_info.Xbar_w + pe_n * num_filter_per_pe * self.model_info.filter_bit
                                    w = bw + xb_w_idx * self.hd_info.Xbar_w + pe_n * num_filter_per_pe * cells_per_filter
                                    h = bh + xb_h_idx * self.hd_info.Xbar_h

                                    # nfilter = w // self.model_info.filter_bit
                                    # nbit = w % self.model_info.filter_bit
                                    nfilter = w // cells_per_filter
                                    start_bit = w % cells_per_filter * self.hd_info.cell_bit_width
                                    # end_bit = start_bit + self.hd_info.cell_bit_width
                                    # if end_bit >= self.model_info.filter_bit:
                                    #     end_bit = self.model_info.filter_bit
                                    # nbit = [i for i in range(start_bit, end_bit)]
                                    nbit = start_bit
                                    ngrid = h

                                    cell_h = bh
                                    cell_w = bw
                                    #self.crossbar_array[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w][cell_h][cell_w] = CrossbarGridMetaData(nlayer, ngrid, nfilter, nbit)
                                    self.crossbar_array[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w][cell_h][cell_w] = [nlayer, ngrid, nfilter, nbit]

                            ## Inputs
                            xbar_inputs = inputs[:, xb_h_idx * self.hd_info.Xbar_h : xb_h_idx * self.hd_info.Xbar_h + block_height].tolist()
                            xbar_column = [i for i in range(block_width)]
                            xbar_row = [i for i in range(block_height)]
                            for inp in xbar_inputs:
                                self.layer_mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w].append(MappingMetaData("fully", nlayer, xbar_row, xbar_column, inp))

                            this_layer_xb_mapping_idx += 1

                # next layer change pe
                used_pe_num = ceil(self.model_info.filter_n[nlayer] / num_filter_per_pe)
                used_xbar_num = used_pe_num * self.num_of_xb_in_pe
                xbar_mapping_idx += used_xbar_num                

            elif self.model_info.layer_list[nlayer].layer_type == "pooling":
                print("Pooling", nlayer)
                o_height = self.model_info.input_h[nlayer] // self.model_info.pooling_h[nlayer]
                o_width = self.model_info.input_w[nlayer] // self.model_info.pooling_w[nlayer]
                inputs = []
                for oh in range(o_height):
                    for ow in range(o_width):
                        for c in range(self.model_info.filter_n[nlayer-1]):
                            nn = []
                            for ph in range(self.model_info.pooling_h[nlayer]):
                                for pw in range(self.model_info.pooling_w[nlayer]):
                                    nn.append([oh*self.model_info.pooling_h[nlayer] + ph, ow*self.model_info.pooling_w[nlayer] + pw, c])
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

                    if pe_n + 1 == used_pe_num:
                        this_input = inputs[pe_n * input_per_pe : ].tolist()
                    else:
                        this_input = inputs[pe_n * input_per_pe : (pe_n+1) * input_per_pe].tolist()
                    xbar_column = [-1]
                    xbar_row = [-1]
                    if this_input:
                        self.layer_mapping_to_pe[rt_h][rt_w][pe_h][pe_w].append(MappingMetaData("pooling", nlayer, xbar_row, xbar_column, this_input))

                xbar_mapping_idx += used_xbar_num

    def __str__(self):
        return str(self.__dict__)
