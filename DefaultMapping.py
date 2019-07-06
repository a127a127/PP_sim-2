from MappingMetaData import MappingMetaData
from CrossbarGridMetaData import CrossbarGridMetaData
import sys
from math import ceil
import numpy as np

class DefaultMapping(object):
    def __init__(self, hd_info, model_info):
        self.model_info = model_info
        self.hd_info = hd_info

        self.layer_list = model_info.layer_list  # conv, pool, conv, ...
        self.layer_length = len(model_info.layer_list)  # how many layer
        self.input_h = [model_info.input_h] # input feature map height (each layer)
        self.input_w = [model_info.input_w] # input feature map width (each layer)
        self.input_c = [model_info.input_c] # input feature map channel (each layer)
        #self.input_number = [] # 會產生幾個output, 也就是多少個input windows # useless?
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
                #self.input_number.append((self.input_h[i] - self.layer_list[i].filter_h + 1) * (self.input_w[i] - self.layer_list[i].filter_w + 1))
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
                #self.input_number.append((self.input_h[i] // self.layer_list[i].pooling_h) * (self.input_w[i] // self.layer_list[i].pooling_w) * (self.input_c[i]))
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
                #self.input_number.append(self.layer_list[i].neuron_n)
            '''
            elif self.layer_list[i].layer_type == "output":
                self.filter_n.append(0)
                self.filter_h.append(0)
                self.filter_w.append(0)
                self.filter_c.append(0)
                self.filter_length.append(0)
                self.pooling_h.append(0)
                self.pooling_w.append(0)
                self.input_h.append(self.filter_n[i])
                self.input_w.append(1)
                self.input_c.append(1)
                self.input_number.append(1)
            '''
        
        # print('self.filter_n, self.filter_h, self.filter_w, self.filter_c, self.filter_length, self.input_h, self.input_w, self.input_c, self.input_number')
        # for i in range(self.layer_length):
        #    print(self.layer_list[i].layer_type)
        #    print(self.filter_n[i], self.filter_h[i], self.filter_w[i], self.filter_c[i], self.filter_length[i], self.input_h[i], self.input_w[i], self.input_c[i], self.input_number[i])

        ## initialize Xbar cell to 0
        ## crossbar_array shape = (PE_idx, CU_idx, Xbar_idx, Xbar_h_idx, Xbar_w_idx)
        self.crossbar_array = []   # 放weights
        self.layer_mapping_to_xbar = [] # inputs
        self.layer_mapping_to_pe = []  # pooling的input

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
                        else:
                            empt = (w+1) * self.hd_info.Xbar_w - matrix_width
                            full = self.hd_info.Xbar_w - empt
                            xbar_column = [i for i in range(full)]
                        self.layer_mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w].append(MappingMetaData("convolution", nlayer, xbar_column, xbar_inputs))

                pe_idx += pe_total_num

                # check result
                # print(self.crossbar_array[0][0][0][0][0][0][1][0][2][0].nbit) # (nlayer, ngrid, nfilter, nbit)
                # print(self.layer_mapping_to_xbar[0][0][0][0][0][1][0][0].xbar_inputs)
                # print(self.layer_mapping_to_xbar[0][0][0][0][0][1][0][0].xbar_column)
        
            elif self.model_info.layer_list[nlayer].layer_type == "pooling":
                pe_idx -= pe_total_num
                o_height = self.input_h[nlayer] // self.pooling_h[nlayer]
                o_width = self.input_w[nlayer] // self.pooling_w[nlayer]
                #print(nlayer, o_height, o_width)
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
                #print(nlayer, inputs)
                l = len(inputs) // pe_total_num
                for pe_n in range(pe_total_num):
                    rt_h = self.pe_mapping_dict[pe_n][0]
                    rt_w = self.pe_mapping_dict[pe_n][1]
                    pe_h = self.pe_mapping_dict[pe_n][2]
                    pe_w = self.pe_mapping_dict[pe_n][3]

                    if pe_n == pe_total_num-1: # last pe deal with the last
                        this_input = inputs[pe_n * l :].tolist()
                    else:
                        this_input = inputs[pe_n * l : (pe_n+1) * l].tolist()
                    xbar_column = [-1]
                    self.layer_mapping_to_pe[rt_h][rt_w][pe_h][pe_w].append(MappingMetaData("pooling", nlayer, xbar_column, this_input))
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
                        self.layer_mapping_to_xbar[rt_h][rt_w][pe_h][pe_w][cu_h][cu_w][xb_h][xb_w].append(MappingMetaData("fully", nlayer, xbar_column, xbar_inputs))
                pe_idx += pe_total_num
        return
    

    def __str__(self):
        return str(self.__dict__)
