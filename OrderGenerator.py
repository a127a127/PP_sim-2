from HardwareMetaData import HardwareMetaData
from configs.ModelConfig import ModelConfig
from Model import Model
from FreeBufferController import FreeBufferController
from PE import PE
from XB import XB
from EventMetaData import EventMetaData
import numpy as np 

class OrderGenerator(object):
    def __init__(self, mapping_information, trace, isFreeBuffer):
        self.model_config = ModelConfig()
        self.model_info = Model(self.model_config)
        self.hd_info = HardwareMetaData()
        self.mapping_information = mapping_information

        self.isFreeBuffer = isFreeBuffer
        if self.isFreeBuffer:
            self.free_buffer_controller = FreeBufferController()

        # mapping
        self.crossbar_array = self.mapping_information.crossbar_array
        self.layer_mapping_to_xbar = self.mapping_information.layer_mapping_to_xbar
        self.layer_mapping_to_pe = self.mapping_information.layer_mapping_to_pe

        self.XB_array = []
        self.cu_traverse_idx = []
        for rty_idx in range(self.hd_info.Router_num_y):
            for rtx_idx in range(self.hd_info.Router_num_x):
                for pey_idx in range(self.hd_info.PE_num_y):
                    for pex_idx in range(self.hd_info.PE_num_x):
                        for cuy_idx in range(self.hd_info.CU_num_y):
                            for cux_idx in range(self.hd_info.CU_num_x):
                                cu_pos = (rty_idx, rtx_idx, pey_idx, pex_idx, cuy_idx, cux_idx)
                                self.cu_traverse_idx.append(cu_pos)
                                for xby_idx in range(self.hd_info.Xbar_num_y):
                                    for xbx_idx in range(self.hd_info.Xbar_num_x):
                                        xb_pos = (rty_idx, rtx_idx, pey_idx, pex_idx, cuy_idx, cux_idx, xby_idx, xbx_idx)
                                        xb = XB(xb_pos)
                                        # weights
                                        xb.crossbar_array = self.crossbar_array[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx][xbx_idx]
                                    
                                        # inputs
                                        for mapping_inp in self.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx][xbx_idx]:
                                            if mapping_inp.eventtype == "convolution":
                                                xb.Convolution.append(mapping_inp)
                                            if mapping_inp.eventtype == "fully":
                                                xb.Fully.append(mapping_inp)
                                        self.XB_array.append(xb)

        self.pe_saa_mat = [] # for dependency
        self.feature_mat = [] # for dependency
        for i in range(self.model_info.layer_length):
            if self.model_info.layer_list[i].layer_type == "convolution":
                self.pe_saa_mat.append(np.zeros((self.model_info.input_number[i], self.model_info.filter_n[i])).tolist())
                if i+1 < self.model_info.layer_length:
                    if self.model_info.layer_list[i+1].layer_type == "fully":
                        self.feature_mat.append(np.zeros((self.model_info.input_h[i+1] * self.model_info.input_w[i+1] * self.model_info.input_c[i+1], 1, 1)).tolist())
                    else:
                        self.feature_mat.append(np.zeros((self.model_info.input_h[i+1], self.model_info.input_w[i+1], self.model_info.input_c[i+1])).tolist())
            elif self.model_info.layer_list[i].layer_type == "pooling":
                self.pe_saa_mat.append([])
                if i+1 < self.model_info.layer_length:
                    if self.model_info.layer_list[i+1].layer_type == "fully":
                        self.feature_mat.append(np.zeros((self.model_info.input_h[i+1] * self.model_info.input_w[i+1] * self.model_info.input_c[i+1], 1, 1)).tolist())
                    else:
                        self.feature_mat.append(np.zeros((self.model_info.input_h[i+1], self.model_info.input_w[i+1], self.model_info.input_c[i+1])).tolist())
            elif self.model_info.layer_list[i].layer_type == "fully":
                self.pe_saa_mat.append(np.zeros((self.model_info.input_number[i], self.model_info.filter_n[i])).tolist())
                if i+1 < self.model_info.layer_length:
                    self.feature_mat.append(np.zeros((self.model_info.filter_n[i], 1, 1)).tolist())
        
        self.Computation_order = []
        self.generate_order()
        if trace:
            self.trace_order()

    def generate_order(self):
        for nlayer in range(self.model_info.layer_length):
            print("Generate layer", nlayer, self.model_info.layer_list[nlayer].layer_type)
            if self.model_info.layer_list[nlayer].layer_type == "convolution":
                ### Event: data_transfer, edram_rd_ir
                for nCU in range(len(self.cu_traverse_idx)):
                    cu_pos = self.cu_traverse_idx[nCU]
                    pe_pos = cu_pos[:-2]
                    rty_idx, rtx_idx= cu_pos[0], cu_pos[1]
                    pey_idx, pex_idx = cu_pos[2], cu_pos[3]
                    cuy_idx, cux_idx = cu_pos[4], cu_pos[5]
                    
                    max_xb_input_len = 0
                    for xby_idx in range(self.hd_info.Xbar_num_y):
                        for xbx_idx in range(self.hd_info.Xbar_num_x):
                            xbar_array_idx = xbx_idx + (xby_idx * self.hd_info.Xbar_num_x) + \
                                            (cux_idx * self.hd_info.Xbar_num) + \
                                            (cuy_idx * self.hd_info.CU_num_x * self.hd_info.Xbar_num) + \
                                            (pex_idx * self.hd_info.CU_num * self.hd_info.Xbar_num) + \
                                            (pey_idx * self.hd_info.PE_num_x * self.hd_info.Xbar_num * self.hd_info.CU_num) + \
                                            (rtx_idx * self.hd_info.PE_num * self.hd_info.CU_num * self.hd_info.Xbar_num) + \
                                            (rty_idx * self.hd_info.Router_num_x * self.hd_info.PE_num * self.hd_info.CU_num * self.hd_info.Xbar_num)
                            num_inp = 0
                            for mapping_inp in self.XB_array[xbar_array_idx].Convolution:
                                if mapping_inp.nlayer == nlayer:
                                    num_inp += 1
                            max_xb_input_len = max(max_xb_input_len, num_inp)
                    
                    for nInp in range(max_xb_input_len):
                        data_feed_to_cu = []
                        for xby_idx in range(self.hd_info.Xbar_num_y):
                            for xbx_idx in range(self.hd_info.Xbar_num_x):
                                xbar_array_idx = xbx_idx + (xby_idx * self.hd_info.Xbar_num_x) + \
                                                (cux_idx * self.hd_info.Xbar_num) + \
                                                (cuy_idx * self.hd_info.CU_num_x * self.hd_info.Xbar_num) + \
                                                (pex_idx * self.hd_info.CU_num * self.hd_info.Xbar_num) + \
                                                (pey_idx * self.hd_info.PE_num_x * self.hd_info.Xbar_num * self.hd_info.CU_num) + \
                                                (rtx_idx * self.hd_info.PE_num * self.hd_info.CU_num * self.hd_info.Xbar_num) + \
                                                (rty_idx * self.hd_info.Router_num_x * self.hd_info.PE_num * self.hd_info.CU_num * self.hd_info.Xbar_num)

                                idx = 0
                                inp = []
                                for mapping_inp in self.XB_array[xbar_array_idx].Convolution:
                                    if mapping_inp.nlayer == nlayer:
                                        if idx == nInp:
                                            inp = mapping_inp.inputs
                                            break
                                        else:
                                            idx += 1
                                for d in inp:
                                    if d not in data_feed_to_cu:
                                        data_feed_to_cu.append(d)

                        eri_preceding_count = 0
                        if nlayer != 0:
                            start_append_idx = len(self.Computation_order)
                            for input_data in data_feed_to_cu:
                                preceding_list = self.feature_mat[nlayer-1][input_data[1]][input_data[2]][input_data[3]] # [h][w][c]
                                #print(preceding_list)
                                if preceding_list != 0.0:
                                    ### Event: data_transfer
                                    for pre_event in preceding_list: 
                                        edram_wr_event = self.Computation_order[pre_event]
                                        data_transfer_source      = edram_wr_event.position_idx
                                        data_transfer_destination = pe_pos
                                        if data_transfer_source != data_transfer_destination:
                                            data_transfer_event_idx = len(self.Computation_order)
                                            edram_wr_event.proceeding_event.append(data_transfer_event_idx)
                                            data_transfer_preceding_count = 1
                                            data_transfer_inputs  = edram_wr_event.inputs # [[h, w, c]]
                                            data_transfer_outputs = edram_wr_event.outputs
                                            event = EventMetaData("data_transfer", [data_transfer_source, data_transfer_destination], data_transfer_preceding_count, [], nlayer-1, data_transfer_inputs, data_transfer_outputs)
                                            self.Computation_order.append(event)

                                            if self.isFreeBuffer:
                                                # input require
                                                pe_id = data_transfer_source[3] + data_transfer_source[2]*self.hd_info.PE_num_x + data_transfer_source[1]*self.hd_info.PE_num + data_transfer_source[0]*self.hd_info.PE_num*self.hd_info.Router_num_x
                                                for d in data_transfer_outputs:
                                                    pos = d[1] + d[0]*self.model_info.input_w[nlayer] + d[2]*self.model_info.input_w[nlayer]*self.model_info.input_h[nlayer] # w + h*width + c*height*width
                                                    self.free_buffer_controller.input_require[pe_id][nlayer][pos] += 1

                            # dependency
                            eri_event_idx = len(self.Computation_order)
                            for input_data in data_feed_to_cu:
                                preceding_list = self.feature_mat[nlayer-1][input_data[1]][input_data[2]][input_data[3]] # [h][w][c]
                                for pre_event in preceding_list:
                                    edram_wr_event = self.Computation_order[pre_event]
                                    data_transfer_source      = edram_wr_event.position_idx
                                    data_transfer_destination = pe_pos
                                    if data_transfer_source == data_transfer_destination:
                                        self.Computation_order[pre_event].proceeding_event.append(eri_event_idx)
                                    eri_preceding_count += 1
                            
                            for idx in range(start_append_idx, eri_event_idx): # dependency (transfer)
                                self.Computation_order[idx].proceeding_event.append(eri_event_idx)

                        eri_event_idx = len(self.Computation_order)
                        eri_position_idx = cu_pos
                        eri_input_sequence = data_feed_to_cu # [[num_input, h, w, c]]
                        eri_output_sequence = data_feed_to_cu
                        event = EventMetaData("edram_rd_ir", eri_position_idx, eri_preceding_count, [], nlayer, eri_input_sequence, eri_output_sequence)
                        self.Computation_order.append(event)

                        if self.isFreeBuffer:
                            # input require
                            pe_id = cu_pos[3] + cu_pos[2]*self.hd_info.PE_num_x + \
                                    cu_pos[1]*self.hd_info.PE_num + cu_pos[0]*self.hd_info.PE_num*self.hd_info.Router_num_x
                            for d in eri_output_sequence:
                                pos = d[2] + d[1]*self.model_info.input_w[nlayer] + d[3]*self.model_info.input_w[nlayer]*self.model_info.input_h[nlayer] # w + h*width + c*height*width
                                self.free_buffer_controller.input_require[pe_id][nlayer][pos] += 1
                ### Event: ou
                        for xby_idx in range(self.hd_info.Xbar_num_y):
                            for xbx_idx in range(self.hd_info.Xbar_num_x):
                                xbar_array_idx = xbx_idx + (xby_idx * self.hd_info.Xbar_num_x) + \
                                                (cux_idx * self.hd_info.Xbar_num) + \
                                                (cuy_idx * self.hd_info.CU_num_x * self.hd_info.Xbar_num) + \
                                                (pex_idx * self.hd_info.CU_num * self.hd_info.Xbar_num) + \
                                                (pey_idx * self.hd_info.PE_num_x * self.hd_info.Xbar_num * self.hd_info.CU_num) + \
                                                (rtx_idx * self.hd_info.PE_num * self.hd_info.CU_num * self.hd_info.Xbar_num) + \
                                                (rty_idx * self.hd_info.Router_num_x * self.hd_info.PE_num * self.hd_info.CU_num * self.hd_info.Xbar_num)

                                idx = 0
                                this_input = []
                                for mapping_inp in self.XB_array[xbar_array_idx].Convolution:
                                    if mapping_inp.nlayer == nlayer:
                                        if idx == nInp:
                                            this_input = mapping_inp
                                            break
                                        else:
                                            idx += 1
                                if not this_input:
                                    continue

                                num_input = this_input.inputs[0][0]

                                xbar_block_h = []
                                xbar_block_w = []
                                index = 0
                                while index < len(this_input.xbar_row): # y-axis
                                    this_block = this_input.xbar_row[index:index + self.hd_info.OU_h]
                                    xbar_block_h.append(this_block)
                                    index += self.hd_info.OU_h
                                index = 0
                                while index < len(this_input.xbar_column): # x-axis 
                                    this_block = this_input.xbar_column[index:index + self.hd_info.OU_w]
                                    xbar_block_w.append(this_block)
                                    index += self.hd_info.OU_w

                                for input_bit in range(self.model_info.input_bit):
                                    for xh in xbar_block_h:
                                        for xw in xbar_block_w:
                                            # OU block
                                            ou_inputs = []
                                            ou_outputs = []  # [[(num_input, input_h, input_w, input_c, input_bit), (nfilter, ngrid, filter_bit)]]
                                            idx = 0
                                            for h in xh:
                                                for w in xw:
                                                    hinput = this_input.inputs[idx][0]
                                                    winput = this_input.inputs[idx][1]
                                                    cinput = this_input.inputs[idx][2]
                                                    
                                                    crossbar_grid = self.XB_array[xbar_array_idx].crossbar_array[h][w]
                                                    filter_nfilter = crossbar_grid.nfilter
                                                    filter_ngrid = crossbar_grid.ngrid
                                                    filter_nbit = crossbar_grid.nbit

                                                    ou_inputs.append([(num_input, hinput, winput, cinput, input_bit)])
                                                    ou_outputs.append([(num_input, hinput, winput, cinput, input_bit), \
                                                                            (filter_nfilter, filter_ngrid, filter_nbit)])
                                                idx += 1
                                            ### add dependency
                                            ou_event_idx = len(self.Computation_order)
                                            self.Computation_order[eri_event_idx].proceeding_event.append(ou_event_idx)
                                            position_idx = self.XB_array[xbar_array_idx].position
                                            preceding_count = 1
                                            event = EventMetaData("ou", position_idx, preceding_count, [], nlayer, ou_inputs, ou_outputs)
                                            self.Computation_order.append(event)              
                ### Event: adc
                                            position_idx = self.XB_array[xbar_array_idx].position
                                            preceding_count = 1
                                            adc_inputs = []
                                            adc_outputs = []

                                            ### add dependency
                                            adc_event_idx = len(self.Computation_order)
                                            self.Computation_order[ou_event_idx].proceeding_event.append(adc_event_idx)

                                            event = EventMetaData("adc", position_idx, preceding_count, [], nlayer, adc_inputs, adc_outputs)
                                            self.Computation_order.append(event)
                ### Event: cu_saa
                                            filter_list = []                            
                                            for column in range(len(xw)): # 同個ou column必須是同一張filter, 只traverse第一個row一次即可
                                                filter_nfilter = ou_outputs[column][1][0]
                                                if filter_nfilter not in filter_list:
                                                    filter_list.append(filter_nfilter)
                                            
                                            for nfilter in filter_list:
                                                cu_saa_inputs = []  #[(input_nbit, filter_nfilter, filter_nbit)] 
                                                for column in range(len(xw)):
                                                    if nfilter == ou_outputs[column][1][0]:
                                                        input_nbit = ou_outputs[0][0][1]
                                                        filter_nbit = ou_outputs[column][1][1]
                                                        cu_saa_inputs.append((num_input, input_nbit, nfilter, filter_nbit))

                                                ### add dependency
                                                cu_saa_event_idx = len(self.Computation_order)
                                                self.Computation_order[adc_event_idx].proceeding_event.append(cu_saa_event_idx)

                                                grid = self.pe_saa_mat[nlayer][num_input][nfilter]
                                                if grid == 0.0:
                                                    self.pe_saa_mat[nlayer][num_input][nfilter] = []

                                                self.pe_saa_mat[nlayer][num_input][nfilter].append(cu_saa_event_idx)
                                                
                                                position_idx = self.XB_array[xbar_array_idx].position[:-2]
                                                preceding_count = 1
                                                cu_saa_outputs = [num_input, nfilter]
                                                event = EventMetaData("cu_saa", position_idx, preceding_count, [], nlayer, cu_saa_inputs, cu_saa_outputs)
                                                self.Computation_order.append(event)
                ### Event: edram_wr, data_transfer (for pe_saa), pe_saa
                windowlen_w = self.model_info.input_w[nlayer] - self.model_info.filter_w[nlayer] + 1 # stride = 1
                windowlen_h = self.model_info.input_h[nlayer] - self.model_info.filter_h[nlayer] + 1 # stride = 1
                for nfilter in range(self.model_info.filter_n[nlayer]):
                    for window_h in range(windowlen_h):
                        for window_w in range(windowlen_w):
                            num_input = window_h * windowlen_w + window_w
                            preceding_list = self.pe_saa_mat[nlayer][num_input][nfilter] 
                            pe_saa_preceding_count = 0 # append pe_saa前有幾個event先append了
                            first_pre_event_idx = preceding_list[0]  # do pe_saa in first pe of preceding cu_saa event
                            do_pe_saa_pos = self.Computation_order[first_pre_event_idx].position_idx[:-2]
                            start_append_idx = len(self.Computation_order)
                ### Event: edram_wr, data_transfer (for pe_saa)
                            preceding_cu = dict()
                            for pre_event_idx in preceding_list:
                                if self.Computation_order[pre_event_idx].position_idx[:-2] != do_pe_saa_pos: # data in other pe
                                    if self.Computation_order[pre_event_idx].position_idx not in preceding_cu:
                                        preceding_cu[self.Computation_order[pre_event_idx].position_idx] = [pre_event_idx]
                                    else:
                                        preceding_cu[self.Computation_order[pre_event_idx].position_idx].append(pre_event_idx)
                            for cu_idx in preceding_cu:
                                edram_wr_event_idx = len(self.Computation_order)
                                for pre_event_idx in preceding_cu[cu_idx]:
                                    self.Computation_order[pre_event_idx].proceeding_event.append(edram_wr_event_idx)
                                pe_idx = cu_idx[0:4]

                                edram_wr_pe_pos  = pe_idx
                                edram_wr_inputs  = [[window_h, window_w, nfilter, "u"]]
                                edram_wr_outputs = [[window_h, window_w, nfilter, "u"]]
                                edram_wr_preceding_count = len(preceding_cu[cu_idx])
                                event = EventMetaData("edram_wr", edram_wr_pe_pos, edram_wr_preceding_count, [edram_wr_event_idx+1], nlayer, edram_wr_inputs, edram_wr_outputs)
                                self.Computation_order.append(event)

                                source_pe_idx = pe_idx
                                transfer_inputs = [[window_h, window_w, nfilter, "u"]]
                                transfer_outputs = [[window_h, window_w, nfilter, "u"]]
                                event = EventMetaData("data_transfer", [source_pe_idx, do_pe_saa_pos], 1, [], nlayer, transfer_inputs, transfer_outputs)
                                self.Computation_order.append(event)

                                pe_saa_preceding_count += 1
                ### Event: pe_saa
                            pe_saa_event_idx = len(self.Computation_order)
                            preceding_cu_idx = list()
                            preceding_tmp_data = list()
                            for pre_event_idx in preceding_list:
                                if self.Computation_order[pre_event_idx].position_idx[:-2] == do_pe_saa_pos: # in same PE
                                    self.Computation_order[pre_event_idx].proceeding_event.append(pe_saa_event_idx)
                                    pe_saa_preceding_count += 1
                                else:
                                    data_transfer_id = self.Computation_order[pre_event_idx].proceeding_event[0] + 1
                                    preceding_tmp_data.append(self.Computation_order[data_transfer_id].outputs[0]) # data transfer output
                                if self.Computation_order[pre_event_idx].position_idx not in preceding_cu_idx:
                                    preceding_cu_idx.append(self.Computation_order[pre_event_idx].position_idx)
                            ### add dependency
                            for idx in range(start_append_idx+1, pe_saa_event_idx, 2):
                                self.Computation_order[idx].proceeding_event.append(pe_saa_event_idx)
                            pe_saa_inputs  = [preceding_cu_idx, preceding_tmp_data]
                            pe_saa_outputs = [[window_h, window_w, nfilter]]
                            event = EventMetaData("pe_saa", do_pe_saa_pos, pe_saa_preceding_count, [], nlayer, pe_saa_inputs, pe_saa_outputs)
                            self.Computation_order.append(event)
                ### Event: activation
                            do_act_pos = do_pe_saa_pos 
                            act_preceding_count = 1
                            act_inputs  = [[window_h, window_w, nfilter]]
                            act_outputs = [[window_h, window_w, nfilter]]
                            ### add dependency
                            act_event_idx = len(self.Computation_order)
                            self.Computation_order[pe_saa_event_idx].proceeding_event.append(act_event_idx)
                            
                            event = EventMetaData("activation", do_act_pos, act_preceding_count, [], nlayer, act_inputs, act_outputs)
                            self.Computation_order.append(event)
                ### Event: edram_wr
                            edram_wr_event_idx = len(self.Computation_order)
                            self.Computation_order[act_event_idx].proceeding_event.append(edram_wr_event_idx)

                            do_edram_wr_pos = do_act_pos
                            edram_wr_preceding_count = 1
                            edram_wr_inputs  = [[window_h, window_w, nfilter]]
                            edram_wr_outputs = [[window_h, window_w, nfilter]]
                                    
                            if nlayer+1 < self.model_info.layer_length:
                                if self.model_info.layer_list[nlayer+1].layer_type != "fully":
                                    if self.feature_mat[nlayer][window_h][window_w][nfilter] == 0.0:
                                        self.feature_mat[nlayer][window_h][window_w][nfilter] = []
                                    self.feature_mat[nlayer][window_h][window_w][nfilter].append(edram_wr_event_idx)
                                else:
                                    seq = window_w + window_h * windowlen_w + nfilter * windowlen_w * windowlen_h
                                    edram_wr_inputs  = [[seq, 0, 0]]
                                    #edram_wr_outputs = [[seq, 0, 0]]
                                    if self.feature_mat[nlayer][window_h * self.model_info.input_w[nlayer+1] + window_w + nfilter * self.model_info.input_h[nlayer+1] * self.model_info.input_w[nlayer+1]][0][0] == 0.0:
                                        self.feature_mat[nlayer][window_h * self.model_info.input_w[nlayer+1] + window_w + nfilter * self.model_info.input_h[nlayer+1] * self.model_info.input_w[nlayer+1]][0][0] = []
                                    self.feature_mat[nlayer][window_h * self.model_info.input_w[nlayer+1] + window_w + nfilter * self.model_info.input_h[nlayer+1] * self.model_info.input_w[nlayer+1]][0][0].append(edram_wr_event_idx)
                            event = EventMetaData("edram_wr", do_edram_wr_pos, edram_wr_preceding_count, [], nlayer, edram_wr_inputs, edram_wr_outputs)
                            self.Computation_order.append(event)
            elif self.model_info.layer_list[nlayer].layer_type == "fully":
                ### Event: data_transfer, edram_rd_ir
                for nCU in range(len(self.cu_traverse_idx)):
                    cu_pos = self.cu_traverse_idx[nCU]
                    pe_pos = cu_pos[:-2]
                    rty_idx, rtx_idx= cu_pos[0], cu_pos[1]
                    pey_idx, pex_idx = cu_pos[2], cu_pos[3]
                    cuy_idx, cux_idx = cu_pos[4], cu_pos[5]
                    
                    max_xb_input_len = 0
                    for xby_idx in range(self.hd_info.Xbar_num_y):
                        for xbx_idx in range(self.hd_info.Xbar_num_x):
                            xbar_array_idx = xbx_idx + (xby_idx * self.hd_info.Xbar_num_x) + \
                                            (cux_idx * self.hd_info.Xbar_num) + \
                                            (cuy_idx * self.hd_info.CU_num_x * self.hd_info.Xbar_num) + \
                                            (pex_idx * self.hd_info.CU_num * self.hd_info.Xbar_num) + \
                                            (pey_idx * self.hd_info.PE_num_x * self.hd_info.Xbar_num * self.hd_info.CU_num) + \
                                            (rtx_idx * self.hd_info.PE_num * self.hd_info.CU_num * self.hd_info.Xbar_num) + \
                                            (rty_idx * self.hd_info.Router_num_x * self.hd_info.PE_num * self.hd_info.CU_num * self.hd_info.Xbar_num)
                            num_inp = 0
                            for mapping_inp in self.XB_array[xbar_array_idx].Fully:
                                if mapping_inp.nlayer == nlayer:
                                    num_inp += 1
                            max_xb_input_len = max(max_xb_input_len, num_inp)
                    
                    for nInp in range(max_xb_input_len):
                        data_feed_to_cu = []
                        for xby_idx in range(self.hd_info.Xbar_num_y):
                            for xbx_idx in range(self.hd_info.Xbar_num_x):
                                xbar_array_idx = xbx_idx + (xby_idx * self.hd_info.Xbar_num_x) + \
                                                (cux_idx * self.hd_info.Xbar_num) + \
                                                (cuy_idx * self.hd_info.CU_num_x * self.hd_info.Xbar_num) + \
                                                (pex_idx * self.hd_info.CU_num * self.hd_info.Xbar_num) + \
                                                (pey_idx * self.hd_info.PE_num_x * self.hd_info.Xbar_num * self.hd_info.CU_num) + \
                                                (rtx_idx * self.hd_info.PE_num * self.hd_info.CU_num * self.hd_info.Xbar_num) + \
                                                (rty_idx * self.hd_info.Router_num_x * self.hd_info.PE_num * self.hd_info.CU_num * self.hd_info.Xbar_num)

                                idx = 0
                                inp = []
                                for mapping_inp in self.XB_array[xbar_array_idx].Fully:
                                    if mapping_inp.nlayer == nlayer:
                                        if idx == nInp:
                                            inp = mapping_inp.inputs
                                            break
                                        else:
                                            idx += 1
                                for d in inp:
                                    if d not in data_feed_to_cu:
                                        data_feed_to_cu.append(d)

                        eri_preceding_count = 0
                        if nlayer != 0:
                            start_append_idx = len(self.Computation_order)
                            
                            for input_data in data_feed_to_cu:
                                preceding_list = self.feature_mat[nlayer-1][input_data[1]][input_data[2]][input_data[3]] # [h][w][c]
                                #print(preceding_list)
                                if preceding_list != 0.0:
                                    ### Event: data_transfer
                                    for pre_event in preceding_list: 
                                        edram_wr_event = self.Computation_order[pre_event]
                                        data_transfer_source      = edram_wr_event.position_idx
                                        data_transfer_destination = pe_pos
                                        if data_transfer_source != data_transfer_destination:
                                            data_transfer_event_idx = len(self.Computation_order)
                                            edram_wr_event.proceeding_event.append(data_transfer_event_idx)
                                            data_transfer_preceding_count = 1
                                            data_transfer_inputs  = edram_wr_event.inputs
                                            data_transfer_outputs = edram_wr_event.outputs
                                            event = EventMetaData("data_transfer", [data_transfer_source, data_transfer_destination], data_transfer_preceding_count, [], nlayer-1, data_transfer_inputs, data_transfer_outputs)
                                            self.Computation_order.append(event)

                                            if self.isFreeBuffer:
                                                # input require
                                                pe_id = data_transfer_source[3] + data_transfer_source[2]*self.hd_info.PE_num_x + data_transfer_source[1]*self.hd_info.PE_num + data_transfer_source[0]*self.hd_info.PE_num*self.hd_info.Router_num_x
                                                if nlayer-1 >= 0 and self.model_info.layer_list[nlayer-1].layer_type != "fully":
                                                    for d in data_transfer_outputs:
                                                        pos = d[1] + d[0]*self.model_info.input_w[nlayer] + d[2]*self.model_info.input_w[nlayer]*self.model_info.input_h[nlayer] # w + h*width + c*height*width
                                                        self.free_buffer_controller.input_require[pe_id][nlayer][pos] += 1
                                                else:
                                                    for d in data_transfer_outputs:
                                                        pos = d[1]
                                                        self.free_buffer_controller.input_require[pe_id][nlayer][pos] += 1
                            # dependency
                            eri_event_idx = len(self.Computation_order)
                            for input_data in data_feed_to_cu:
                                preceding_list = self.feature_mat[nlayer-1][input_data[1]][input_data[2]][input_data[3]] # [h][w][c]
                                for pre_event in preceding_list:
                                    edram_wr_event = self.Computation_order[pre_event]
                                    data_transfer_source      = edram_wr_event.position_idx
                                    data_transfer_destination = pe_pos
                                    if data_transfer_source == data_transfer_destination:
                                        self.Computation_order[pre_event].proceeding_event.append(eri_event_idx)
                                    eri_preceding_count += 1
                            
                            for idx in range(start_append_idx, eri_event_idx): # dependency (transfer)
                                self.Computation_order[idx].proceeding_event.append(eri_event_idx)

                        eri_event_idx = len(self.Computation_order)
                        eri_position_idx = cu_pos
                        eri_input_sequence = data_feed_to_cu
                        eri_output_sequence = data_feed_to_cu
                        event = EventMetaData("edram_rd_ir", eri_position_idx, eri_preceding_count, [], nlayer, eri_input_sequence, eri_output_sequence)
                        self.Computation_order.append(event)

                        if self.isFreeBuffer:
                            # input require
                            pe_id = cu_pos[3] + cu_pos[2]*self.hd_info.PE_num_x + \
                                    cu_pos[1]*self.hd_info.PE_num + cu_pos[0]*self.hd_info.PE_num*self.hd_info.Router_num_x
                            for d in eri_output_sequence:
                                pos = d[1]
                                self.free_buffer_controller.input_require[pe_id][nlayer][pos] += 1
                ### Event: ou
                        for xby_idx in range(self.hd_info.Xbar_num_y):
                            for xbx_idx in range(self.hd_info.Xbar_num_x):
                                xbar_array_idx = xbx_idx + (xby_idx * self.hd_info.Xbar_num_x) + \
                                                (cux_idx * self.hd_info.Xbar_num) + \
                                                (cuy_idx * self.hd_info.CU_num_x * self.hd_info.Xbar_num) + \
                                                (pex_idx * self.hd_info.CU_num * self.hd_info.Xbar_num) + \
                                                (pey_idx * self.hd_info.PE_num_x * self.hd_info.Xbar_num * self.hd_info.CU_num) + \
                                                (rtx_idx * self.hd_info.PE_num * self.hd_info.CU_num * self.hd_info.Xbar_num) + \
                                                (rty_idx * self.hd_info.Router_num_x * self.hd_info.PE_num * self.hd_info.CU_num * self.hd_info.Xbar_num)

                                idx = 0
                                this_input = []
                                for mapping_inp in self.XB_array[xbar_array_idx].Fully:
                                    if mapping_inp.nlayer == nlayer:
                                        if idx == nInp:
                                            this_input = mapping_inp
                                            break
                                        else:
                                            idx += 1
                                if not this_input:
                                    continue

                                num_input = this_input.inputs[0][0]

                                xbar_block_h = []
                                xbar_block_w = []
                                index = 0
                                while index < len(this_input.xbar_row): # y-axis
                                    this_block = this_input.xbar_row[index:index + self.hd_info.OU_h]
                                    xbar_block_h.append(this_block)
                                    index += self.hd_info.OU_h
                                index = 0
                                while index < len(this_input.xbar_column): # x-axis 
                                    this_block = this_input.xbar_column[index:index + self.hd_info.OU_w]
                                    xbar_block_w.append(this_block)
                                    index += self.hd_info.OU_w
                            

                                for input_bit in range(self.model_info.input_bit):
                                    for xh in xbar_block_h:
                                        for xw in xbar_block_w:
                                            # OU block
                                            ou_inputs = []
                                            ou_outputs = []  # [[(input_h, input_w, input_c, input_bit), (nfilter, ngrid, filter_bit)]]
                                            idx = 0
                                            for h in xh:
                                                for w in xw:
                                                    hinput = this_input.inputs[idx][0]
                                                    winput = this_input.inputs[idx][1]
                                                    cinput = this_input.inputs[idx][2]
                                                    
                                                    crossbar_grid = self.XB_array[xbar_array_idx].crossbar_array[h][w]
                                                    filter_nfilter = crossbar_grid.nfilter
                                                    filter_ngrid = crossbar_grid.ngrid
                                                    filter_nbit = crossbar_grid.nbit

                                                    ou_inputs.append([(num_input, hinput, winput, cinput, input_bit)])
                                                    ou_outputs.append([(num_input, hinput, winput, cinput, input_bit), \
                                                                            (filter_nfilter, filter_ngrid, filter_nbit)])
                                                idx += 1
                                            ### add dependency
                                            ou_event_idx = len(self.Computation_order)
                                            self.Computation_order[eri_event_idx].proceeding_event.append(ou_event_idx)
                                            position_idx = self.XB_array[xbar_array_idx].position
                                            preceding_count = 1
                                            event = EventMetaData("ou", position_idx, preceding_count, [], nlayer, ou_inputs, ou_outputs)
                                            self.Computation_order.append(event)
                ### Event: adc
                                            position_idx = self.XB_array[xbar_array_idx].position
                                            preceding_count = 1
                                            adc_inputs = []
                                            adc_outputs = []

                                            ### add dependency
                                            adc_event_idx = len(self.Computation_order)
                                            self.Computation_order[ou_event_idx].proceeding_event.append(adc_event_idx)

                                            event = EventMetaData("adc", position_idx, preceding_count, [], nlayer, adc_inputs, adc_outputs)
                                            self.Computation_order.append(event)
                ### Event: cu_saa
                                            filter_list = []
                                            for column in range(len(xw)): # 同個ou column必須是同一張filter, 只traverse第一個row一次即可
                                                filter_nfilter = ou_outputs[column][1][0]
                                                if filter_nfilter not in filter_list:
                                                    filter_list.append(filter_nfilter)
                                            
                                            for nfilter in filter_list:
                                                cu_saa_inputs = []  #[(input_nbit, filter_nfilter, filter_nbit)] 
                                                for column in range(len(xw)):
                                                    if nfilter == ou_outputs[column][1][0]:
                                                        input_nbit = ou_outputs[0][0][1]
                                                        filter_nbit = ou_outputs[column][1][1]
                                                        cu_saa_inputs.append((num_input, input_nbit, nfilter, filter_nbit))

                                                ### add dependency
                                                cu_saa_event_idx = len(self.Computation_order)
                                                self.Computation_order[adc_event_idx].proceeding_event.append(cu_saa_event_idx)

                                                grid = self.pe_saa_mat[nlayer][num_input][nfilter]
                                                if grid == 0.0:
                                                    self.pe_saa_mat[nlayer][num_input][nfilter] = []

                                                self.pe_saa_mat[nlayer][num_input][nfilter].append(cu_saa_event_idx)
                                                
                                                position_idx = self.XB_array[xbar_array_idx].position[:-2]
                                                preceding_count = 1
                                                cu_saa_outputs = [num_input, nfilter]
                                                event = EventMetaData("cu_saa", position_idx, preceding_count, [], nlayer, cu_saa_inputs, cu_saa_outputs)
                                                self.Computation_order.append(event)
                ### Event: edram_wr, data_transfer (for pe_saa), pe_saa
                for nfilter in range(self.model_info.filter_n[nlayer]):
                    num_input = 0
                    grid = self.pe_saa_mat[nlayer][num_input][nfilter]
                    if grid == 0.0:
                        self.pe_saa_mat[nlayer][num_input][nfilter] = []
                    preceding_list = self.pe_saa_mat[nlayer][num_input][nfilter]
                    
                    pe_saa_preceding_count = 0 # append pe_saa前有幾個event先append了
                    first_pre_event_idx = preceding_list[0]  # do pe_saa in first pe of preceding cu_saa event
                    do_pe_saa_pos = self.Computation_order[first_pre_event_idx].position_idx[:-2]
                    
                    start_append_idx = len(self.Computation_order)
                ### Event: edram_wr, data_transfer (for pe_saa)
                    preceding_cu = dict()
                    for pre_event_idx in preceding_list:
                        if self.Computation_order[pre_event_idx].position_idx[:-2] != do_pe_saa_pos: # data in other pe
                            if self.Computation_order[pre_event_idx].position_idx not in preceding_cu:
                                preceding_cu[self.Computation_order[pre_event_idx].position_idx] = [pre_event_idx]
                            else:
                                preceding_cu[self.Computation_order[pre_event_idx].position_idx].append(pre_event_idx)
                                
                    for cu_idx in preceding_cu:
                        edram_wr_event_idx = len(self.Computation_order)
                        for pre_event_idx in preceding_cu[cu_idx]:
                            self.Computation_order[pre_event_idx].proceeding_event.append(edram_wr_event_idx)
                        pe_idx = cu_idx[0:4]
                        edram_wr_pe_pos  = pe_idx
                        edram_wr_inputs  = [[nfilter, 0, 0, "u"]]
                        edram_wr_outputs = [[nfilter, 0, 0, "u"]]
                        edram_wr_preceding_count = len(preceding_cu[cu_idx])
                        event = EventMetaData("edram_wr", edram_wr_pe_pos, edram_wr_preceding_count, [edram_wr_event_idx+1], nlayer, edram_wr_inputs, edram_wr_outputs)
                        self.Computation_order.append(event)

                        source_pe_idx = pe_idx
                        transfer_inputs = [[nfilter, 0, 0, "u"]]
                        transfer_outputs = [[nfilter, 0, 0, "u"]]
                        event = EventMetaData("data_transfer", [source_pe_idx, do_pe_saa_pos], 1, [], nlayer, transfer_inputs, transfer_outputs)
                        self.Computation_order.append(event)
                            
                        pe_saa_preceding_count += 1
                ### Event: pe_saa
                    pe_saa_event_idx = len(self.Computation_order)
                    preceding_cu_idx = list()
                    preceding_tmp_data = list()
                    for pre_event_idx in preceding_list:
                        if self.Computation_order[pre_event_idx].position_idx[:-2] == do_pe_saa_pos: # in same PE
                            self.Computation_order[pre_event_idx].proceeding_event.append(pe_saa_event_idx)
                            pe_saa_preceding_count += 1
                        else:
                            data_transfer_id = self.Computation_order[pre_event_idx].proceeding_event[0] + 1
                            preceding_tmp_data.append(self.Computation_order[data_transfer_id].outputs[0]) # data transfer output
                        if self.Computation_order[pre_event_idx].position_idx not in preceding_cu_idx:
                            preceding_cu_idx.append(self.Computation_order[pre_event_idx].position_idx)

                    ### add dependency
                    for idx in range(start_append_idx+1, pe_saa_event_idx, 2):
                        self.Computation_order[idx].proceeding_event.append(pe_saa_event_idx)
                    
                    pe_saa_inputs  = [preceding_cu_idx, preceding_tmp_data]
                    pe_saa_outputs = [[nfilter, 0, 0]]
                    event = EventMetaData("pe_saa", do_pe_saa_pos, pe_saa_preceding_count, [], nlayer, pe_saa_inputs, pe_saa_outputs)
                    self.Computation_order.append(event)
                ### Event: activation
                    do_act_pos = do_pe_saa_pos 
                    act_preceding_count = 1
                    act_inputs  = [[0, 0, nfilter]]
                    act_outputs = [[0, 0, nfilter]]
                    ### add dependency
                    act_event_idx = len(self.Computation_order)
                    self.Computation_order[pe_saa_event_idx].proceeding_event.append(act_event_idx)
                    
                    event = EventMetaData("activation", do_act_pos, act_preceding_count, [], nlayer, act_inputs, act_outputs)
                    self.Computation_order.append(event)
                ### Event: edram_wr
                    edram_wr_event_idx = len(self.Computation_order)
                    self.Computation_order[act_event_idx].proceeding_event.append(edram_wr_event_idx)

                    do_edram_wr_pos = do_act_pos
                    edram_wr_preceding_count = 1
                    edram_wr_inputs  = [[nfilter, 0, 0]]
                    edram_wr_outputs = [[nfilter, 0, 0]]
                    event = EventMetaData("edram_wr", do_edram_wr_pos, edram_wr_preceding_count, [], nlayer, edram_wr_inputs, edram_wr_outputs)
                    self.Computation_order.append(event) 

                    if nlayer+1 < self.model_info.layer_length:
                        if self.feature_mat[nlayer][nfilter][0][0] == 0.0:
                            self.feature_mat[nlayer][nfilter][0][0] = []
                        self.feature_mat[nlayer][nfilter][0][0].append(edram_wr_event_idx)
            elif self.model_info.layer_list[nlayer].layer_type == "pooling":
                for rty_idx in range(self.hd_info.Router_num_y):
                    for rtx_idx in range(self.hd_info.Router_num_x):
                        for pey_idx in range(self.hd_info.PE_num_y):
                            for pex_idx in range(self.hd_info.PE_num_x):
                                pe_pos = (rty_idx, rtx_idx, pey_idx, pex_idx)
                                for mapping_data in self.layer_mapping_to_pe[rty_idx][rtx_idx][pey_idx][pex_idx]:
                                    if mapping_data.nlayer == nlayer:
                ### Event: edram_rd_pool, data_transfer
                                        for inputs in mapping_data.inputs:
                                            # [[h, w, c], [h, w, c], ...]
                                            erp_preceding_count = 0
                                            start_append_idx = len(self.Computation_order)
                                            for input_data in inputs:
                                                preceding_list = self.feature_mat[nlayer-1][input_data[0]][input_data[1]][input_data[2]]
                                                if preceding_list == 0:
                                                    print("preceding_list == 0 error.")
                                                    exit()
                                                else:
                                                    for pre_event in preceding_list:
                                                        edram_wr_event = self.Computation_order[pre_event]
                                                        data_transfer_source      = edram_wr_event.position_idx
                                                        data_transfer_destination = pe_pos
                                                        if data_transfer_source != data_transfer_destination:
                                                            ## Event: data_transfer
                                                            data_transfer_event_idx = len(self.Computation_order)
                                                            edram_wr_event.proceeding_event.append(data_transfer_event_idx)
                                                            data_transfer_preceding_count = 1
                                                            data_transfer_inputs  = edram_wr_event.outputs
                                                            data_transfer_outputs = edram_wr_event.outputs
                                                            event = EventMetaData("data_transfer", [data_transfer_source, data_transfer_destination], data_transfer_preceding_count, [], nlayer-1, data_transfer_inputs, data_transfer_outputs)
                                                            self.Computation_order.append(event)
                                        
                                            # dependency
                                            erp_event_idx = len(self.Computation_order)
                                            for input_data in inputs:
                                                preceding_list = self.feature_mat[nlayer-1][input_data[0]][input_data[1]][input_data[2]]
                                                if preceding_list == 0:
                                                    print("preceding_list == 0 error.")
                                                    exit()
                                                else:
                                                    for pre_event in preceding_list:
                                                        edram_wr_event = self.Computation_order[pre_event]
                                                        data_transfer_source      = edram_wr_event.position_idx
                                                        data_transfer_destination = pe_pos
                                                        if data_transfer_source == data_transfer_destination:
                                                            self.Computation_order[pre_event].proceeding_event.append(erp_event_idx)
                                                        erp_preceding_count += 1

                                            for idx in range(start_append_idx, erp_event_idx):
                                                self.Computation_order[idx].proceeding_event.append(erp_event_idx)

                                            erp_event_idx = len(self.Computation_order)
                                            erp_position_idx = pe_pos
                                            erp_input_sequence = inputs
                                            erp_output_sequence = inputs
                                            event = EventMetaData("edram_rd_pool", erp_position_idx, erp_preceding_count, [], nlayer, erp_input_sequence, erp_output_sequence)
                                            self.Computation_order.append(event)

                                            if self.isFreeBuffer:
                                                # input require
                                                pe_id = pe_pos[3] + pe_pos[2]*self.hd_info.PE_num_x + \
                                                        pe_pos[1]*self.hd_info.PE_num + pe_pos[0]*self.hd_info.PE_num*self.hd_info.Router_num_x
                                                for d in erp_output_sequence:
                                                    pos = d[1] + d[0]*self.model_info.input_w[nlayer] + d[2]*self.model_info.input_w[nlayer]*self.model_info.input_h[nlayer] # w + h*width + c*height*width
                                                    self.free_buffer_controller.input_require[pe_id][nlayer][pos] += 1
                ### Event: pooling
                                            pool_event_index = len(self.Computation_order)
                                            self.Computation_order[erp_event_idx].proceeding_event.append(pool_event_index)

                                            pool_position_idx = pe_pos
                                            pool_preceding_count = 1
                                            pool_input_sequence = erp_input_sequence
                                            pool_output_sequence = [[pool_input_sequence[0][0] // self.model_info.pooling_h[nlayer], pool_input_sequence[0][1] // self.model_info.pooling_w[nlayer], pool_input_sequence[0][2]]]

                                            event = EventMetaData("pooling", pool_position_idx, pool_preceding_count, [], nlayer, pool_input_sequence, pool_output_sequence)
                                            self.Computation_order.append(event)
                ### Event: edram_wr
                                            edram_wr_event_idx = len(self.Computation_order)
                                            self.Computation_order[pool_event_index].proceeding_event.append(edram_wr_event_idx)

                                            edram_wr_position_idx = pe_pos
                                            edram_wr_preceding_count = 1
                                            edram_wr_input_sequence  = pool_output_sequence
                                            edram_wr_output_sequence = pool_output_sequence

                                            if nlayer+1 < len(self.model_info.layer_list):
                                                if self.model_info.layer_list[nlayer+1].layer_type != "fully":
                                                    if self.feature_mat[nlayer][edram_wr_output_sequence[0][0]][edram_wr_output_sequence[0][1]][edram_wr_output_sequence[0][2]] == 0.0:
                                                        self.feature_mat[nlayer][edram_wr_output_sequence[0][0]][edram_wr_output_sequence[0][1]][edram_wr_output_sequence[0][2]] = []
                                                    self.feature_mat[nlayer][edram_wr_output_sequence[0][0]][edram_wr_output_sequence[0][1]][edram_wr_output_sequence[0][2]].append(edram_wr_event_idx)
                                                else:
                                                    o_height = self.model_info.input_h[nlayer] // self.model_info.pooling_h[nlayer]
                                                    o_width = self.model_info.input_w[nlayer] // self.model_info.pooling_w[nlayer]
                                                    edram_wr_input_sequence = [[(pool_input_sequence[0][0] // self.model_info.pooling_h[nlayer]) * o_width + pool_input_sequence[0][1] // self.model_info.pooling_w[nlayer] + pool_input_sequence[0][2] * o_width * o_height, 0, 0]]
                                                    #edram_wr_output_sequence = edram_wr_input_sequence
                                                    if self.feature_mat[nlayer][pool_input_sequence[0][0] * self.model_info.input_w[nlayer+1] + pool_input_sequence[0][1] + pool_input_sequence[0][2] * self.model_info.input_h[nlayer+1] * self.model_info.input_w[nlayer+1]][0][0] == 0.0:
                                                        self.feature_mat[nlayer][pool_input_sequence[0][0] * self.model_info.input_w[nlayer+1] + pool_input_sequence[0][1] + pool_input_sequence[0][2] * self.model_info.input_h[nlayer+1] * self.model_info.input_w[nlayer+1]][0][0] = []
                                                    self.feature_mat[nlayer][pool_input_sequence[0][0] * self.model_info.input_w[nlayer+1] + pool_input_sequence[0][1] + pool_input_sequence[0][2] * self.model_info.input_h[nlayer+1] * self.model_info.input_w[nlayer+1]][0][0].append(edram_wr_event_idx)
                                            event = EventMetaData("edram_wr", edram_wr_position_idx, edram_wr_preceding_count, [], nlayer, edram_wr_input_sequence, edram_wr_output_sequence)
                                            self.Computation_order.append(event)
        print('Order generated!')

    def trace_order(self):
        edram_rd_ir_ctr = 0
        ou_ctr = 0
        adc_ctr = 0
        cu_saa_ctr = 0
        pe_saa_ctr = 0
        activation_ctr = 0
        pooling_ctr = 0
        edram_wr_ctr = 0
        edram_rd_pool_ctr = 0
        data_transfer_ctr = 0
        for e in self.Computation_order:
            t = e.event_type
            if t == "edram_rd_ir":
                edram_rd_ir_ctr += 1
            elif t == "ou":
                ou_ctr += 1
            elif t == "adc":
                adc_ctr += 1
            elif t == "cu_saa":
                cu_saa_ctr += 1
            elif t == "pe_saa":
                pe_saa_ctr += 1
            elif t == "activation":
                activation_ctr += 1
            elif t == "edram_wr":
                edram_wr_ctr += 1
            elif t == "edram_rd_pool":
                edram_rd_pool_ctr += 1
            elif t == "pooling":
                pooling_ctr += 1
            elif t == "data_transfer":
                data_transfer_ctr += 1
            else:
                print("event type error:", t)

        print("edram_rd_ir_ctr", edram_rd_ir_ctr)
        print("ou_ctr", ou_ctr)
        print("adc_ctr", adc_ctr)
        print("cu_saa_ctr", cu_saa_ctr)
        print("pe_saa_ctr", pe_saa_ctr)
        print("activation_ctr", activation_ctr)
        print("edram_wr_ctr", edram_wr_ctr)
        print("edram_rd_pool_ctr", edram_rd_pool_ctr)
        print("data_transfer_ctr", data_transfer_ctr)

        for e in self.Computation_order:
            print(self.Computation_order.index(e), e)

    def __str__(self):
        return str(self.__dict__)