from HardwareMetaData import HardwareMetaData as HW
from ModelConfig import ModelConfig
from Model import Model
from FreeBufferController import FreeBufferController
from PE import PE
from EventMetaData import EventMetaData
import numpy as np
import math
import collections

class OrderGenerator(object):
    def __init__(self, mapping_information, trace):
        model_config = ModelConfig()
        self.model_info = Model(model_config)
        self.mp_info = mapping_information
        #self.free_buffer_controller = FreeBufferController()

       #---紀錄每一筆feature map data會被哪些PE用到---#
        self.fm_data_used_pe_idx = []
        for i in range(self.model_info.layer_length+1):
            arr = []
            for h in range(self.model_info.input_h[i] * self.model_info.input_w[i] * self.model_info.input_c[i]):
                arr.append(set())
            self.fm_data_used_pe_idx.append(arr)
        
        for nlayer in range(self.model_info.layer_length):
            if self.model_info.layer_list[nlayer].layer_type == "convolution":
                strides = self.model_info.strides[nlayer]
                pad = self.model_info.pad[nlayer]
                o_height = self.model_info.input_h[nlayer+1]
                o_width = self.model_info.input_w[nlayer+1]
                for window_h in range(o_height):
                    for window_w in range(o_width):
                       #---此window的data---#
                        input_vector_data = []
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
                        for pe_pos in self.mp_info.layer_used_pe[nlayer]:
                            rty_idx, rtx_idx = pe_pos[0], pe_pos[1]
                            pey_idx, pex_idx = pe_pos[2], pe_pos[3]
                            pe_inp_vetor_pos = set()
                            for cu_idx in range(HW().CU_num):
                                for xb_idx in range(HW().Xbar_num):
                                    xbar_inputs = self.mp_info.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx][nlayer]
                                    if xbar_inputs:
                                        inp_vector_position = xbar_inputs[0].inputs
                                        for pos in inp_vector_position:
                                            pe_inp_vetor_pos.add(pos)
                            
                            for pos in pe_inp_vetor_pos:
                                if input_vector_data[pos] != 0:
                                    h = input_vector_data[pos][1]
                                    w = input_vector_data[pos][2]
                                    c = input_vector_data[pos][3]
                                    n = w + h * self.model_info.input_w[nlayer] + c * self.model_info.input_w[nlayer] * self.model_info.input_h[nlayer]
                                    self.fm_data_used_pe_idx[nlayer][n].add(pe_pos)
                            
            elif self.model_info.layer_list[nlayer].layer_type == "fully":
                input_vector_data = []
                for h in range(self.model_info.filter_length[nlayer]):
                    input_vector_data.append((nlayer, h, 0, 0))

                for pe_pos in self.mp_info.layer_used_pe[nlayer]:
                    rty_idx, rtx_idx = pe_pos[0], pe_pos[1]
                    pey_idx, pex_idx = pe_pos[2], pe_pos[3]
                    pe_inp_vetor_pos = set()
                    for cu_idx in range(HW().CU_num):
                        for xb_idx in range(HW().Xbar_num):
                            xbar_inputs = self.mp_info.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx][nlayer]
                            if xbar_inputs:
                                inp_vector_position = xbar_inputs[0].inputs
                                for pos in inp_vector_position:
                                    pe_inp_vetor_pos.add(pos)
                            
                    for pos in pe_inp_vetor_pos:
                        if input_vector_data[pos] != 0:
                            n = input_vector_data[pos][1]
                            self.fm_data_used_pe_idx[nlayer][n].add(pe_pos)
                            
            elif self.model_info.layer_list[nlayer].layer_type == "pooling":
                for pe_pos in self.mp_info.layer_used_pe[nlayer]:
                    rty_idx, rtx_idx = pe_pos[0], pe_pos[1]
                    pey_idx, pex_idx = pe_pos[2], pe_pos[3]

                    pe_inputs = self.mp_info.mapping_to_pe[rty_idx][rtx_idx][pey_idx][pex_idx][nlayer]
                    for pool_inp in pe_inputs:
                        for data in pool_inp[1]:
                            h = data[1]
                            w = data[2]
                            c = data[3]
                            n = w + h * self.model_info.input_w[nlayer] + c * self.model_info.input_w[nlayer] * self.model_info.input_h[nlayer]           
                            self.fm_data_used_pe_idx[nlayer][n].add(pe_pos)
       #------------------------------------------#
       
       #---紀錄feature map data transfer event的index---#
        self.fm_data_transfer_event_idx = []
        for nlayer in range(self.model_info.layer_length):
            self.fm_data_transfer_event_idx.append([])
            for i in range(self.model_info.input_h[nlayer+1]*self.model_info.input_w[nlayer+1]*self.model_info.input_c[nlayer+1]):
                self.fm_data_transfer_event_idx[nlayer].append(dict()) # {PE: transfer_event_idx}
       #-----------------------------------------------#  

        self.Computation_order = []
        self.generate_order()
        if trace:
            self.print_order()
       
    def generate_order(self):
        for nlayer in range(self.model_info.layer_length):
            print("Generate layer", nlayer, self.model_info.layer_list[nlayer].layer_type)
            if self.model_info.layer_list[nlayer].layer_type == "convolution":
               #----決定每個filter的aggregator----#
                pe_filter_processing = dict() # {PE1:{"act":[f1, f2], "transfer": {des_pe1:[f3, f4], des_pe2:[f5,f6]}, "aggregate":[f7]}, PE2: ...}
                
                # 每個filter在哪一些PEs算
                pe_operate_filter    = dict() # {PE1: {filter1, filter2, ...}, PE2: {filter2, filter3, ...}
                for pe_pos in self.mp_info.layer_used_pe[nlayer]:
                    rty_idx, rtx_idx = pe_pos[0], pe_pos[1]
                    pey_idx, pex_idx = pe_pos[2], pe_pos[3]
                    operate_filter = set() # 此pe有算到哪些filter
                    for cu_idx in range(HW().CU_num):
                        for xb_idx in range(HW().Xbar_num):
                            xbar_inputs = self.mp_info.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx][nlayer]
                            if xbar_inputs:
                                filter_list = xbar_inputs[0].Filters # 此xbar 會計算到的 filter
                                for f in filter_list:
                                    operate_filter.add(f)
                    pe_operate_filter[pe_pos] = operate_filter
                
                # 每個filter的aggregator pe, 和non aggregator pe
                filter_aggregator = [] # {"aggregator": PE1, "other": [PE2, PE3]}
                for f in range(self.model_info.filter_n[nlayer]):
                    aggregator_dict = {"aggregator": 0, "non": []}
                    for pe_pos in pe_operate_filter:
                        if f in pe_operate_filter[pe_pos]:
                            if aggregator_dict["aggregator"] == 0:
                                aggregator_dict["aggregator"] = pe_pos
                            else:
                                aggregator_dict["non"].append(pe_pos)
                    filter_aggregator.append(aggregator_dict)
                
                # 完成pe_filter_processing
                for pe_pos in pe_operate_filter:
                    pe_filter_processing[pe_pos] = {"act": [], "transfer": dict(), "aggregate": []}
                    operate_filter = pe_operate_filter[pe_pos]
                    for f in operate_filter:
                        if not filter_aggregator[f]["non"]: # []
                            pe_filter_processing[pe_pos]["act"].append(f)
                        elif pe_pos == filter_aggregator[f]["aggregator"]: # pe為f的aggregator
                            pe_filter_processing[pe_pos]["aggregate"].append(f)
                        else: # 要transfer至別的pe aggregate
                            des_pe = filter_aggregator[f]["aggregator"]
                            if des_pe not in pe_filter_processing[pe_pos]["transfer"]:
                                pe_filter_processing[pe_pos]["transfer"][des_pe] = [f]
                            else:
                                pe_filter_processing[pe_pos]["transfer"][des_pe].append(f)
               #--------------------------------#
                
                strides = self.model_info.strides[nlayer]
                pad = self.model_info.pad[nlayer]
                o_height = self.model_info.input_h[nlayer+1]
                o_width = self.model_info.input_w[nlayer+1]
                for window_h in range(o_height):
                    for window_w in range(o_width):
                       # 兩個用來記錄event index的字典 # for dependency
                        pe_saa_event_dict = dict()
                        wr_and_transfer_event_dict = dict()
                        for pe_pos in self.mp_info.layer_used_pe[nlayer]:
                            pe_saa_event_dict[pe_pos] = []
                            wr_and_transfer_event_dict[pe_pos] = []
                       #===準備此window的data===#
                        input_vector_data = []
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
                       #======================#

                       #===生一個window的events===#
                        for pe_pos in self.mp_info.layer_used_pe[nlayer]: # loop此layer會map到的PE # 一次把一個PE的event生完
                            rty_idx, rtx_idx = pe_pos[0], pe_pos[1]
                            pey_idx, pex_idx = pe_pos[2], pe_pos[3]

                            for cu_idx in range(HW().CU_num): # check all CU in PE
                               #---Event: edram_rd_ir---#
                                eri_event_idx = len(self.Computation_order)
                                eri_position_idx = (rty_idx, rtx_idx, pey_idx, pex_idx, cu_idx)
                                
                                # 準備edram read一次讀多少data
                                edram_read_data = []
                                cu_inp_vetor_pos = set()
                                for xb_idx in range(HW().Xbar_num):
                                    xbar_inputs = self.mp_info.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx][nlayer]
                                    if xbar_inputs:
                                        inp_vector_position = xbar_inputs[0].inputs
                                        for pos in inp_vector_position:
                                            cu_inp_vetor_pos.add(pos)
                                if not cu_inp_vetor_pos:
                                    # 此CU沒有任何input vector要做
                                    continue
                                for pos in cu_inp_vetor_pos:
                                    if input_vector_data[pos] != 0: # pading值不要
                                        edram_read_data.append(input_vector_data[pos])

                                # dependency: transfer -> edram_rd_ir
                                pre_event = set()
                                if nlayer != 0:
                                    for data in edram_read_data:
                                        h = data[1]
                                        w = data[2]
                                        c = data[3]
                                        pos = w + h * self.model_info.input_w[nlayer] + \
                                            c * self.model_info.input_w[nlayer] * self.model_info.input_h[nlayer]
                                        transfer_event_idx = self.fm_data_transfer_event_idx[nlayer-1][pos][pe_pos]
                                        pre_event.add(transfer_event_idx)
                                        if eri_event_idx not in self.Computation_order[transfer_event_idx].proceeding_event:
                                            self.Computation_order[transfer_event_idx].proceeding_event.append(eri_event_idx)

                                if nlayer == 0:
                                    eri_preceding_count = 0
                                else:
                                    eri_preceding_count = len(pre_event)
                                
                                eri_inputs  = edram_read_data
                                eri_outputs = 0
                                event = EventMetaData("edram_rd_ir", eri_position_idx, eri_preceding_count, [eri_event_idx+1], nlayer, eri_inputs, eri_outputs)
                                self.Computation_order.append(event)

                                
                                ### input requirement
                                # pe_id = pex_idx + pey_idx*HW().PE_num_x + \
                                #         rtx_idx*HW().PE_num + rty_idx*HW().PE_num*HW().Router_num_x
                                # for data in edram_read_data:
                                #     pos = data[2] + data[1]*self.model_info.input_w[nlayer] + data[3]*self.model_info.input_w[nlayer]*self.model_info.input_h[nlayer] # w + h*width + c*height*width
                                #     self.free_buffer_controller.input_require[pe_id][nlayer][pos] += 1
                               #------------------------#

                               #---Event: cu_operation---#
                                cu_operation_event_idx = len(self.Computation_order)
                                cu_op_position_idx = (rty_idx, rtx_idx, pey_idx, pex_idx, cu_idx)
                                preceding_count = 1

                                # 一個cu operation內有多少ou要做
                                num_ou_in_xb = dict() # {XB1: 4, XB2: 4}
                                max_ou = 0
                                for xb_idx in range(HW().Xbar_num):
                                    xbar_inputs = self.mp_info.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx][nlayer]
                                    if xbar_inputs:
                                        inp = xbar_inputs[0]
                                        num_ou_h = math.ceil(len(inp.inputs)/ HW().OU_h)
                                        num_ou_w = math.ceil(inp.Cols / HW().OU_w)
                                        num_ou = num_ou_h * num_ou_w * self.model_info.input_bit
                                        num_ou_in_xb[xb_idx] = num_ou
                                        max_ou = max(num_ou, max_ou)
                                cu_op_inputs  = max_ou
                                cu_op_outputs = num_ou_in_xb

                                event = EventMetaData("cu_operation", cu_op_position_idx, preceding_count, [cu_operation_event_idx+1], nlayer, cu_op_inputs, cu_op_outputs)
                                self.Computation_order.append(event)
                               #-------------------------#

                               #---Event: pe_saa---#
                                pe_saa_event_idx = len(self.Computation_order)
                                pe_saa_position_idx = (rty_idx, rtx_idx, pey_idx, pex_idx)
                                preceding_count = 1

                                # Shift and add 要做多少次
                                cu_operate_filter = set()
                                for xb_idx in range(HW().Xbar_num):
                                    xbar_inputs = self.mp_info.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx][nlayer]
                                    if xbar_inputs:
                                        filter_list = xbar_inputs[0].Filters # 此xbar 會計算到的 filter
                                        for f in filter_list:
                                            cu_operate_filter.add(f)
                                saa_amount = len(cu_operate_filter)
                                pe_saa_inputs  = saa_amount

                                pe_saa_outputs = 0
                                event = EventMetaData("pe_saa", pe_saa_position_idx, preceding_count, [], nlayer, pe_saa_inputs, pe_saa_outputs)
                                self.Computation_order.append(event)
                                pe_saa_event_dict[pe_pos].append(pe_saa_event_idx)
                               #-------------------#

                           # 1. 在此PE做activation, edram_write, transfer
                            if pe_filter_processing[pe_pos]["act"]:
                               #---Event: activation---#
                                act_event_idx = len(self.Computation_order)
                                act_position_idx = (rty_idx, rtx_idx, pey_idx, pex_idx)
                                act_preceding_count = len(pe_saa_event_dict[pe_pos])
                                act_amount = len(pe_filter_processing[pe_pos]["act"])
                                act_inputs  = act_amount
                                act_outputs = 0
                                event = EventMetaData("activation", act_position_idx, act_preceding_count, [], nlayer, act_inputs, act_outputs)
                                self.Computation_order.append(event)
                                for event_idx in pe_saa_event_dict[pe_pos]: # dependency
                                    self.Computation_order[event_idx].proceeding_event.append(act_event_idx)
                               #-----------------------#
                               
                                if nlayer == self.model_info.layer_length-1: # 最後一層直接寫到eDRAM
                                   #---Event: edram_wr---#
                                    wr_event_idx = len(self.Computation_order)
                                    wr_position_idx = (rty_idx, rtx_idx, pey_idx, pex_idx)
                                    wr_preceding_count = 1
                                    wr_inputs  = 0

                                    edram_write_data = []
                                    for f in pe_filter_processing[pe_pos]["act"]:
                                        edram_write_data.append((nlayer+1, window_h, window_w, f))
                                    wr_outputs = edram_write_data
                                
                                    event = EventMetaData("edram_wr", wr_position_idx, wr_preceding_count, [], nlayer, wr_inputs, wr_outputs)
                                    self.Computation_order.append(event)
                                    self.Computation_order[act_event_idx].proceeding_event.append(wr_event_idx) # dependency
                                   #---------------------#

                                else: # 不是最後一層要生data transfer event傳到下一層所在PE
                                    des_pe_dict = dict() # {PE1: [data1, data2], PE2: [data1, data3]}
                                    for f in pe_filter_processing[pe_pos]["act"]:
                                        pos = window_w + window_h * self.model_info.input_w[nlayer+1] + \
                                                f * self.model_info.input_w[nlayer+1] * self.model_info.input_h[nlayer+1]
                                        if self.model_info.layer_list[nlayer+1].layer_type != "fully":
                                            data = (nlayer+1, window_h, window_w, f)
                                        else:
                                            data = (nlayer+1, pos, 0, 0)
                                        des_pe_set = self.fm_data_used_pe_idx[nlayer+1][pos]
                                        for des_pe in des_pe_set:
                                            if des_pe in des_pe_dict:
                                                des_pe_dict[des_pe].append(data)
                                            else:
                                                des_pe_dict[des_pe] = [data]
                                    
                                    for des_pe in des_pe_dict: # 一個目的地PE生一個data transfer event
                                       #---Event: data_transfer---# 
                                        transfer_event_idx = len(self.Computation_order)
                                        data_transfer_src = (rty_idx, rtx_idx, pey_idx, pex_idx)
                                        data_transfer_des = des_pe
                                        transfer_position_idx = [data_transfer_src, data_transfer_des]
                                        transfer_preceding_count = 1
                                        transfer_inputs  = 0
                                        transfer_outputs = des_pe_dict[des_pe]
                                        event = EventMetaData("data_transfer", transfer_position_idx, transfer_preceding_count, [], nlayer, transfer_inputs, transfer_outputs)
                                        self.Computation_order.append(event)
                                        self.Computation_order[act_event_idx].proceeding_event.append(transfer_event_idx)
                                        
                                        # 先記錄data transfer event的index, 在生下一層的eDRAM read才能接dependency
                                        for data in transfer_outputs:
                                            if self.model_info.layer_list[nlayer+1].layer_type != "fully":
                                                f = data[3]
                                                pos = window_w + window_h * self.model_info.input_w[nlayer+1] + \
                                                    f * self.model_info.input_w[nlayer+1] * self.model_info.input_h[nlayer+1]
                                            else:
                                                pos = data[1]
                                            self.fm_data_transfer_event_idx[nlayer][pos][data_transfer_des] = transfer_event_idx
                                       #--------------------------#

                           # 2. 要傳到別的pe做activation 要生transfer
                            if pe_filter_processing[pe_pos]["transfer"]:
                                # pe_filter_processing: {PE1:{"act":[f1, f2], "transfer": {des_pe1:[f3, f4], des_pe2:[f5,f6]}, "aggregate":[f7]}, PE2: ...}
                                for des_pe in pe_filter_processing[pe_pos]["transfer"]:
                                   #---Event: data_transfer---#
                                    transfer_event_idx = len(self.Computation_order)
                                    data_transfer_src = (rty_idx, rtx_idx, pey_idx, pex_idx)
                                    data_transfer_des = des_pe
                                    transfer_position_idx = [data_transfer_src, data_transfer_des]
                                    transfer_preceding_count = len(pe_saa_event_dict[pe_pos])
                                    transfer_inputs  = 0
                                    transfer_outputs = []
                                    for f in pe_filter_processing[pe_pos]["transfer"][des_pe]:
                                        transfer_outputs.append((nlayer, window_h, window_w, f, pe_pos))
                                    event = EventMetaData("data_transfer", transfer_position_idx, transfer_preceding_count, [], nlayer, transfer_inputs, transfer_outputs)
                                    self.Computation_order.append(event)
                                    for event_idx in pe_saa_event_dict[pe_pos]: # dependency
                                        self.Computation_order[event_idx].proceeding_event.append(transfer_event_idx)
                                    
                                    wr_and_transfer_event_dict[des_pe].append(transfer_event_idx) # for dependency
                                   #--------------------------#

                           # 3. aggregator
                            if pe_filter_processing[pe_pos]["aggregate"]:
                               #---Event: edram_wr---#
                                wr_event_idx = len(self.Computation_order)
                                wr_position_idx = (rty_idx, rtx_idx, pey_idx, pex_idx)
                                wr_preceding_count = len(pe_saa_event_dict[pe_pos])
                                wr_inputs  = 0
                                
                                edram_write_data = []
                                # pe_filter_processing: {PE1:{"act":[f1, f2], "transfer": {des_pe1:[f3, f4], des_pe2:[f5,f6]}, "aggregate":[f7]}, PE2: ...}
                                for f in pe_filter_processing[pe_pos]["aggregate"]:
                                    edram_write_data.append((nlayer, window_h, window_w, f, pe_pos))
                                wr_outputs = edram_write_data

                                event = EventMetaData("edram_wr", wr_position_idx, wr_preceding_count, [], nlayer, wr_inputs, wr_outputs)
                                self.Computation_order.append(event)

                                for event_idx in pe_saa_event_dict[pe_pos]: # dependency
                                    self.Computation_order[event_idx].proceeding_event.append(wr_event_idx)
                                wr_and_transfer_event_dict[pe_pos].append(wr_event_idx) # for dependency
                               #---------------------#
                               
                        for pe_pos in self.mp_info.layer_used_pe[nlayer]:
                            rty_idx, rtx_idx = pe_pos[0], pe_pos[1]
                            pey_idx, pex_idx = pe_pos[2], pe_pos[3]

                           # 4. 剩下需要跨PE做 S+A, activation, edram write 的event
                            if pe_filter_processing[pe_pos]["aggregate"]:
                               #---Event: edram_rd---#
                                eri_event_idx = len(self.Computation_order)
                                eri_position_idx = (rty_idx, rtx_idx, pey_idx, pex_idx)
                                eri_preceding_count = len(wr_and_transfer_event_dict[pe_pos])
                                    
                                # 準備edram read一次讀多少data
                                edram_read_data = []
                                for pre_event_idx in wr_and_transfer_event_dict[pe_pos]:
                                    pre_event = self.Computation_order[pre_event_idx]
                                    for data in pre_event.outputs:
                                        edram_read_data.append(data)
                                    
                                eri_inputs  = edram_read_data
                                eri_outputs = 0
                                event = EventMetaData("edram_rd", eri_position_idx, eri_preceding_count, [eri_event_idx+1], nlayer, eri_inputs, eri_outputs)
                                self.Computation_order.append(event)

                                for event_idx in wr_and_transfer_event_dict[pe_pos]:
                                    self.Computation_order[event_idx].proceeding_event.append(eri_event_idx)
                                ### input requirement
                                #
                               #------------------------#

                               #---Event: pe_saa---#
                                pe_saa_event_idx = len(self.Computation_order)
                                pe_saa_position_idx = (rty_idx, rtx_idx, pey_idx, pex_idx)
                                preceding_count = 1

                                # Shift and add 要做多少次
                                saa_amount = len(wr_and_transfer_event_dict[pe_pos]) * len(pe_filter_processing[pe_pos]["aggregate"])
                                pe_saa_inputs  = saa_amount

                                pe_saa_outputs = 0
                                event = EventMetaData("pe_saa", pe_saa_position_idx, preceding_count, [pe_saa_event_idx+1], nlayer, pe_saa_inputs, pe_saa_outputs)
                                self.Computation_order.append(event)
                               #-------------------#

                               #---Event: activation---#
                                act_event_idx = len(self.Computation_order)
                                act_position_idx = (rty_idx, rtx_idx, pey_idx, pex_idx)
                                act_preceding_count = 1
                                act_amount = len(pe_filter_processing[pe_pos]["aggregate"])
                                act_inputs  = act_amount
                                act_outputs = 0
                                event = EventMetaData("activation", act_position_idx, act_preceding_count, [], nlayer, act_inputs, act_outputs)
                                self.Computation_order.append(event)
                               #-----------------------#

                                if nlayer == self.model_info.layer_length-1: # 最後一層直接寫到eDRAM
                                   #---Event: edram_wr---#
                                    wr_event_idx = len(self.Computation_order)
                                    wr_position_idx = (rty_idx, rtx_idx, pey_idx, pex_idx)
                                    wr_preceding_count = 1
                                    wr_inputs  = 0
                                
                                    edram_write_data = []
                                    for f in pe_filter_processing[pe_pos]["aggregate"]:
                                        edram_write_data.append((nlayer+1, window_h, window_w, f))
                                    wr_outputs = edram_write_data

                                    event = EventMetaData("edram_wr", wr_position_idx, wr_preceding_count, [wr_event_idx+1], nlayer, wr_inputs, wr_outputs)
                                    self.Computation_order.append(event)
                                    self.Computation_order[act_event_idx].proceeding_event.append(wr_event_idx) # dependency
                                   #---------------------#

                                else: # 不是最後一層要生data transfer event傳到下一層所在PE
                                    des_pe_dict = dict() # {PE1: [data1, data2], PE2: [data1, data3]}
                                    for f in pe_filter_processing[pe_pos]["aggregate"]:
                                        pos = window_w + window_h * self.model_info.input_w[nlayer+1] + \
                                                f * self.model_info.input_w[nlayer+1] * self.model_info.input_h[nlayer+1]
                                        if self.model_info.layer_list[nlayer+1].layer_type != "fully":
                                            data = (nlayer+1, window_h, window_w, f)
                                        else:
                                            data = (nlayer+1, pos, 0, 0)
                                        des_pe_set = self.fm_data_used_pe_idx[nlayer+1][pos]
                                        for des_pe in des_pe_set:
                                            if des_pe in des_pe_dict:
                                                des_pe_dict[des_pe].append(data)
                                            else:
                                                des_pe_dict[des_pe] = [data]
                                    
                                    for des_pe in des_pe_dict: # 一個目的地PE生一個data transfer event
                                       #---Event: data_transfer---#
                                        transfer_event_idx = len(self.Computation_order)
                                        data_transfer_src = (rty_idx, rtx_idx, pey_idx, pex_idx)
                                        data_transfer_des = des_pe
                                        transfer_position_idx = [data_transfer_src, data_transfer_des]
                                        transfer_preceding_count = 1
                                        transfer_inputs  = 0
                                        transfer_outputs = des_pe_dict[des_pe]
                                        event = EventMetaData("data_transfer", transfer_position_idx, transfer_preceding_count, [], nlayer, transfer_inputs, transfer_outputs)
                                        self.Computation_order.append(event)
                                        self.Computation_order[act_event_idx].proceeding_event.append(transfer_event_idx)
                                        
                                        # 先記錄data transfer event的index, 在生下一層的eDRAM read才能接dependency
                                        for data in transfer_outputs:
                                            if self.model_info.layer_list[nlayer+1].layer_type != "fully":
                                                f = data[3]
                                                pos = window_w + window_h * self.model_info.input_w[nlayer+1] + \
                                                    f * self.model_info.input_w[nlayer+1] * self.model_info.input_h[nlayer+1]
                                            else:
                                                pos = data[1]
                                            self.fm_data_transfer_event_idx[nlayer][pos][data_transfer_des] = transfer_event_idx
                                       #--------------------------#  
                       #=======================#

            elif self.model_info.layer_list[nlayer].layer_type == "fully":
               #----決定每個filter的aggregator----# # 和conv一樣
                pe_filter_processing = dict() # {PE1:{"act":[f1, f2], "transfer": {des_pe1:[f3, f4], des_pe2:[f5,f6]}, "aggregate":[f7]}, PE2: ...}
                
                # 每個filter在哪一些PEs算
                pe_operate_filter    = dict() # {PE1: {filter1, filter2, ...}, PE2: {filter2, filter3, ...}
                for pe_pos in self.mp_info.layer_used_pe[nlayer]:
                    rty_idx, rtx_idx = pe_pos[0], pe_pos[1]
                    pey_idx, pex_idx = pe_pos[2], pe_pos[3]
                    operate_filter = set() # 此pe有算到哪些filter
                    for cu_idx in range(HW().CU_num):
                        for xb_idx in range(HW().Xbar_num):
                            xbar_inputs = self.mp_info.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx][nlayer]
                            if xbar_inputs:
                                filter_list = xbar_inputs[0].Filters # 此xbar 會計算到的 filter
                                for f in filter_list:
                                    operate_filter.add(f)
                    pe_operate_filter[pe_pos] = operate_filter
                
                # 每個filter的aggregator pe, 和non aggregator pe
                filter_aggregator = [] # {"aggregator": PE1, "other": [PE2, PE3]}
                for f in range(self.model_info.filter_n[nlayer]):
                    aggregator_dict = {"aggregator": 0, "non": []}
                    for pe_pos in pe_operate_filter:
                        if f in pe_operate_filter[pe_pos]:
                            if aggregator_dict["aggregator"] == 0:
                                aggregator_dict["aggregator"] = pe_pos
                            else:
                                aggregator_dict["non"].append(pe_pos)
                    filter_aggregator.append(aggregator_dict)
                
                # 完成pe_filter_processing
                for pe_pos in pe_operate_filter:
                    pe_filter_processing[pe_pos] = {"act": [], "transfer": dict(), "aggregate": []}
                    operate_filter = pe_operate_filter[pe_pos]
                    for f in operate_filter:
                        if not filter_aggregator[f]["non"]: # []
                            pe_filter_processing[pe_pos]["act"].append(f)
                        elif pe_pos == filter_aggregator[f]["aggregator"]: # pe為f的aggregator
                            pe_filter_processing[pe_pos]["aggregate"].append(f)
                        else: # 要transfer至別的pe aggregate
                            des_pe = filter_aggregator[f]["aggregator"]
                            if des_pe not in pe_filter_processing[pe_pos]["transfer"]:
                                pe_filter_processing[pe_pos]["transfer"][des_pe] = [f]
                            else:
                                pe_filter_processing[pe_pos]["transfer"][des_pe].append(f)
               #--------------------------------#
                
               # 兩個用來記錄event index的字典 # for dependency
                pe_saa_event_dict = dict()
                wr_and_transfer_event_dict = dict()
                for pe_pos in self.mp_info.layer_used_pe[nlayer]:
                    pe_saa_event_dict[pe_pos] = []
                    wr_and_transfer_event_dict[pe_pos] = []
               #===準備data===#
                input_vector_data = []
                for h in range(self.model_info.filter_length[nlayer]):
                    input_vector_data.append((nlayer, h, 0, 0))
               #=============#

               #===生event===#
                for pe_pos in self.mp_info.layer_used_pe[nlayer]: # loop此layer會map到的PE # 一次把一個PE的event生完
                    rty_idx, rtx_idx = pe_pos[0], pe_pos[1]
                    pey_idx, pex_idx = pe_pos[2], pe_pos[3]

                    for cu_idx in range(HW().CU_num): # check all CU in PE
                       #---Event: edram_rd_ir---#
                        eri_event_idx = len(self.Computation_order)
                                
                        # 準備edram read一次讀多少data
                        edram_read_data = []
                        cu_inp_vetor_pos = set()
                        for xb_idx in range(HW().Xbar_num):
                            xbar_inputs = self.mp_info.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx][nlayer]
                            if xbar_inputs:
                                inp_vector_position = xbar_inputs[0].inputs
                                for pos in inp_vector_position:
                                    cu_inp_vetor_pos.add(pos)
                        if not cu_inp_vetor_pos:
                            # 此CU沒有任何input vector要做
                            continue
                        for pos in cu_inp_vetor_pos:
                            edram_read_data.append(input_vector_data[pos])

                        # dependency: transfer -> edram_rd_ir
                        pre_event = set()
                        if nlayer != 0:
                            for data in edram_read_data:
                                pos = data[1] # h
                                transfer_event_idx = self.fm_data_transfer_event_idx[nlayer-1][pos][pe_pos]
                                pre_event.add(transfer_event_idx)
                                if eri_event_idx not in self.Computation_order[transfer_event_idx].proceeding_event:
                                    self.Computation_order[transfer_event_idx].proceeding_event.append(eri_event_idx)
                        eri_position_idx = (rty_idx, rtx_idx, pey_idx, pex_idx, cu_idx)

                        if nlayer == 0:
                            eri_preceding_count = 0
                        else:
                            eri_preceding_count = len(pre_event)
                                
                        eri_inputs  = edram_read_data
                        eri_outputs = 0
                        event = EventMetaData("edram_rd_ir", eri_position_idx, eri_preceding_count, [eri_event_idx+1], nlayer, eri_inputs, eri_outputs)
                        self.Computation_order.append(event)
                       #------------------------#

                       #---Event: cu_operation---#
                        cu_operation_event_idx = len(self.Computation_order)
                        cu_op_position_idx = (rty_idx, rtx_idx, pey_idx, pex_idx, cu_idx)
                        preceding_count = 1

                        # 一個cu operation內有多少ou要做
                        num_ou_in_xb = dict() # {XB1: 4, XB2: 4}
                        max_ou = 0
                        for xb_idx in range(HW().Xbar_num):
                            xbar_inputs = self.mp_info.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx][nlayer]
                            if xbar_inputs:
                                inp = xbar_inputs[0]
                                num_ou_h = math.ceil(len(inp.inputs)/ HW().OU_h)
                                num_ou_w = math.ceil(inp.Cols / HW().OU_w)
                                num_ou = num_ou_h * num_ou_w * self.model_info.input_bit
                                num_ou_in_xb[xb_idx] = num_ou
                                max_ou = max(num_ou, max_ou)
                        cu_op_inputs  = max_ou
                        cu_op_outputs = num_ou_in_xb

                        event = EventMetaData("cu_operation", cu_op_position_idx, preceding_count, [cu_operation_event_idx+1], nlayer, cu_op_inputs, cu_op_outputs)
                        self.Computation_order.append(event)
                       #-------------------------#

                       #---Event: pe_saa---#
                        pe_saa_event_idx = len(self.Computation_order)
                        pe_saa_position_idx = (rty_idx, rtx_idx, pey_idx, pex_idx)
                        preceding_count = 1

                        # Shift and add 要做多少次
                        cu_operate_filter = set()
                        for xb_idx in range(HW().Xbar_num):
                            xbar_inputs = self.mp_info.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx][nlayer]
                            if xbar_inputs:
                                filter_list = xbar_inputs[0].Filters # 此xbar 會計算到的 filter
                                for f in filter_list:
                                    cu_operate_filter.add(f)
                        saa_amount = len(cu_operate_filter)
                        pe_saa_inputs  = saa_amount

                        pe_saa_outputs = 0
                        event = EventMetaData("pe_saa", pe_saa_position_idx, preceding_count, [], nlayer, pe_saa_inputs, pe_saa_outputs)
                        self.Computation_order.append(event)
                        pe_saa_event_dict[pe_pos].append(pe_saa_event_idx) # dependency: pe_saa -> act, write, transfer
                       #-------------------#

                   # 1. 在此PE做activation, edram_write, transfer
                    if pe_filter_processing[pe_pos]["act"]:
                       #---Event: activation---#
                        act_event_idx = len(self.Computation_order)
                        act_position_idx = (rty_idx, rtx_idx, pey_idx, pex_idx)
                        act_preceding_count = len(pe_saa_event_dict[pe_pos])
                        act_amount = len(pe_filter_processing[pe_pos]["act"])
                        act_inputs  = act_amount
                        act_outputs = 0
                        event = EventMetaData("activation", act_position_idx, act_preceding_count, [], nlayer, act_inputs, act_outputs)
                        self.Computation_order.append(event)
                        for event_idx in pe_saa_event_dict[pe_pos]: # dependency: pe_saa -> act
                            self.Computation_order[event_idx].proceeding_event.append(act_event_idx)
                       #-----------------------#
                               
                        if nlayer == self.model_info.layer_length-1: # 最後一層直接寫到eDRAM
                           #---Event: edram_wr---#
                            wr_event_idx = len(self.Computation_order)
                            wr_position_idx = (rty_idx, rtx_idx, pey_idx, pex_idx)
                            wr_preceding_count = 1
                            wr_inputs  = 0

                            edram_write_data = []
                            for f in pe_filter_processing[pe_pos]["act"]:
                                edram_write_data.append((nlayer+1, f, 0, 0))
                            wr_outputs = edram_write_data
                        
                            event = EventMetaData("edram_wr", wr_position_idx, wr_preceding_count, [], nlayer, wr_inputs, wr_outputs)
                            self.Computation_order.append(event)
                            self.Computation_order[act_event_idx].proceeding_event.append(wr_event_idx) # dependency
                           #---------------------#

                        else: # 不是最後一層要生data transfer event傳到下一層所在PE
                            des_pe_dict = dict() # {PE1: [data1, data2], PE2: [data1, data3]}
                            for f in pe_filter_processing[pe_pos]["act"]:
                                data = (nlayer+1, f, 0, 0)
                                pos = f
                                des_pe_set = self.fm_data_used_pe_idx[nlayer+1][pos]
                                for des_pe in des_pe_set:
                                    if des_pe in des_pe_dict:
                                        des_pe_dict[des_pe].append(data)
                                    else:
                                        des_pe_dict[des_pe] = [data]
                            
                            for des_pe in des_pe_dict: # 一個目的地PE生一個data transfer event
                               #---Event: data_transfer---# 
                                transfer_event_idx = len(self.Computation_order)
                                data_transfer_src = (rty_idx, rtx_idx, pey_idx, pex_idx)
                                data_transfer_des = des_pe
                                transfer_position_idx = [data_transfer_src, data_transfer_des]
                                transfer_preceding_count = 1
                                transfer_inputs  = 0
                                transfer_outputs = des_pe_dict[des_pe]
                                event = EventMetaData("data_transfer", transfer_position_idx, transfer_preceding_count, [], nlayer, transfer_inputs, transfer_outputs)
                                self.Computation_order.append(event)
                                self.Computation_order[act_event_idx].proceeding_event.append(transfer_event_idx)
                                
                                # 先記錄data transfer event的index, 在生下一層的eDRAM read才能接dependency
                                for data in transfer_outputs:
                                    pos = data[1] # f
                                    self.fm_data_transfer_event_idx[nlayer][pos][data_transfer_des] = transfer_event_idx
                               #--------------------------#

                   # 2. 要傳到別的pe做activation 要生transfer
                    if pe_filter_processing[pe_pos]["transfer"]:
                        # pe_filter_processing: {PE1:{"act":[f1, f2], "transfer": {des_pe1:[f3, f4], des_pe2:[f5,f6]}, "aggregate":[f7]}, PE2: ...}
                        for des_pe in pe_filter_processing[pe_pos]["transfer"]:
                           #---Event: data_transfer---#
                            transfer_event_idx = len(self.Computation_order)
                            data_transfer_src = (rty_idx, rtx_idx, pey_idx, pex_idx)
                            data_transfer_des = des_pe
                            transfer_position_idx = [data_transfer_src, data_transfer_des]
                            transfer_preceding_count = len(pe_saa_event_dict[pe_pos])
                            transfer_inputs  = 0
                            transfer_outputs = []
                            for f in pe_filter_processing[pe_pos]["transfer"][des_pe]:
                                transfer_outputs.append((nlayer, f, 0, 0, pe_pos))
                            event = EventMetaData("data_transfer", transfer_position_idx, transfer_preceding_count, [], nlayer, transfer_inputs, transfer_outputs)
                            self.Computation_order.append(event)
                            for event_idx in pe_saa_event_dict[pe_pos]: # dependency
                                self.Computation_order[event_idx].proceeding_event.append(transfer_event_idx)
                            
                            wr_and_transfer_event_dict[des_pe].append(transfer_event_idx) # for dependency
                           #--------------------------#

                   # 3. aggregator
                    if pe_filter_processing[pe_pos]["aggregate"]:
                       #---Event: edram_wr---#
                        wr_event_idx = len(self.Computation_order)
                        wr_position_idx = (rty_idx, rtx_idx, pey_idx, pex_idx)
                        wr_preceding_count = len(pe_saa_event_dict[pe_pos])
                        wr_inputs  = 0
                        
                        edram_write_data = []
                        # pe_filter_processing: {PE1:{"act":[f1, f2], "transfer": {des_pe1:[f3, f4], des_pe2:[f5,f6]}, "aggregate":[f7]}, PE2: ...}
                        for f in pe_filter_processing[pe_pos]["aggregate"]:
                            edram_write_data.append((nlayer, f, 0, 0, pe_pos))
                        wr_outputs = edram_write_data

                        event = EventMetaData("edram_wr", wr_position_idx, wr_preceding_count, [], nlayer, wr_inputs, wr_outputs)
                        self.Computation_order.append(event)

                        for event_idx in pe_saa_event_dict[pe_pos]: # dependency
                            self.Computation_order[event_idx].proceeding_event.append(wr_event_idx)
                        wr_and_transfer_event_dict[pe_pos].append(wr_event_idx) # for dependency
                       #---------------------#
                               
                for pe_pos in self.mp_info.layer_used_pe[nlayer]:
                    rty_idx, rtx_idx = pe_pos[0], pe_pos[1]
                    pey_idx, pex_idx = pe_pos[2], pe_pos[3]
                   # 4. 剩下需要跨PE做 S+A, activation, edram write 的event
                    if pe_filter_processing[pe_pos]["aggregate"]:
                       #---Event: edram_rd---#
                        eri_event_idx = len(self.Computation_order)
                        eri_position_idx = (rty_idx, rtx_idx, pey_idx, pex_idx)
                        eri_preceding_count = len(wr_and_transfer_event_dict[pe_pos])
                            
                        # 準備edram read一次讀多少data
                        edram_read_data = []
                        for pre_event_idx in wr_and_transfer_event_dict[pe_pos]:
                            pre_event = self.Computation_order[pre_event_idx]
                            for data in pre_event.outputs:
                                edram_read_data.append(data)
                            
                        eri_inputs  = edram_read_data
                        eri_outputs = 0
                        event = EventMetaData("edram_rd", eri_position_idx, eri_preceding_count, [eri_event_idx+1], nlayer, eri_inputs, eri_outputs)
                        self.Computation_order.append(event)

                        for event_idx in wr_and_transfer_event_dict[pe_pos]:
                            self.Computation_order[event_idx].proceeding_event.append(eri_event_idx)
                        ### input requirement
                        #
                       #------------------------#

                       #---Event: pe_saa---#
                        pe_saa_event_idx = len(self.Computation_order)
                        pe_saa_position_idx = (rty_idx, rtx_idx, pey_idx, pex_idx)
                        preceding_count = 1

                        # Shift and add 要做多少次
                        saa_amount = len(wr_and_transfer_event_dict[pe_pos]) * len(pe_filter_processing[pe_pos]["aggregate"])
                        pe_saa_inputs  = saa_amount

                        pe_saa_outputs = 0
                        event = EventMetaData("pe_saa", pe_saa_position_idx, preceding_count, [pe_saa_event_idx+1], nlayer, pe_saa_inputs, pe_saa_outputs)
                        self.Computation_order.append(event)
                       #-------------------#

                       #---Event: activation---#
                        act_event_idx = len(self.Computation_order)
                        act_position_idx = (rty_idx, rtx_idx, pey_idx, pex_idx)
                        act_preceding_count = 1
                        act_amount = len(pe_filter_processing[pe_pos]["aggregate"])
                        act_inputs  = act_amount
                        act_outputs = 0
                        event = EventMetaData("activation", act_position_idx, act_preceding_count, [], nlayer, act_inputs, act_outputs)
                        self.Computation_order.append(event)
                       #-----------------------#

                        if nlayer == self.model_info.layer_length-1: # 最後一層直接寫到eDRAM
                           #---Event: edram_wr---#
                            wr_event_idx = len(self.Computation_order)
                            wr_position_idx = (rty_idx, rtx_idx, pey_idx, pex_idx)
                            wr_preceding_count = 1
                            wr_inputs  = 0
                        
                            edram_write_data = []
                            for f in pe_filter_processing[pe_pos]["aggregate"]:
                                edram_write_data.append((nlayer+1, f, 0, 0))
                            wr_outputs = edram_write_data

                            event = EventMetaData("edram_wr", wr_position_idx, wr_preceding_count, [wr_event_idx+1], nlayer, wr_inputs, wr_outputs)
                            self.Computation_order.append(event)
                            self.Computation_order[act_event_idx].proceeding_event.append(wr_event_idx) # dependency
                           #---------------------#

                        else: # 不是最後一層要生data transfer event傳到下一層所在PE
                            des_pe_dict = dict() # {PE1: [data1, data2], PE2: [data1, data3]}
                            for f in pe_filter_processing[pe_pos]["aggregate"]:
                                data = (nlayer+1, f, 0, 0)
                                pos = f
                                des_pe_set = self.fm_data_used_pe_idx[nlayer+1][pos]
                                for des_pe in des_pe_set:
                                    if des_pe in des_pe_dict:
                                        des_pe_dict[des_pe].append(data)
                                    else:
                                        des_pe_dict[des_pe] = [data]
                            
                            for des_pe in des_pe_dict: # 一個目的地PE生一個data transfer event
                               #---Event: data_transfer---#
                                transfer_event_idx = len(self.Computation_order)
                                data_transfer_src = (rty_idx, rtx_idx, pey_idx, pex_idx)
                                data_transfer_des = des_pe
                                transfer_position_idx = [data_transfer_src, data_transfer_des]
                                transfer_preceding_count = 1
                                transfer_inputs  = 0
                                transfer_outputs = des_pe_dict[des_pe]
                                event = EventMetaData("data_transfer", transfer_position_idx, transfer_preceding_count, [], nlayer, transfer_inputs, transfer_outputs)
                                self.Computation_order.append(event)
                                self.Computation_order[act_event_idx].proceeding_event.append(transfer_event_idx)
                                
                                # 先記錄data transfer event的index, 在生下一層的eDRAM read才能接dependency
                                for data in transfer_outputs:
                                    pos = data[1] # f
                                    self.fm_data_transfer_event_idx[nlayer][pos][data_transfer_des] = transfer_event_idx
                               #--------------------------#    
               #=============#

            elif self.model_info.layer_list[nlayer].layer_type == "pooling":
                for pe_pos in self.mp_info.layer_used_pe[nlayer]:
                    rty_idx, rtx_idx = pe_pos[0], pe_pos[1]
                    pey_idx, pex_idx = pe_pos[2], pe_pos[3]

                    pe_inputs = self.mp_info.mapping_to_pe[rty_idx][rtx_idx][pey_idx][pex_idx][nlayer]
                   #---Event: edram_rd---#
                    eri_event_idx = len(self.Computation_order)
                    eri_position_idx = (rty_idx, rtx_idx, pey_idx, pex_idx)
                    
                    edram_read_data = set()
                    for pool_inp in pe_inputs:
                        for data in pool_inp[1]:
                            edram_read_data.add(data)
                    edram_read_data = list(edram_read_data)

                    # dependency: transfer -> edram_rd
                    pre_event = set()
                    if nlayer != 0:
                        for data in edram_read_data:
                            h = data[1]
                            w = data[2]
                            c = data[3]
                            pos = w + h * self.model_info.input_w[nlayer] + \
                                c * self.model_info.input_w[nlayer] * self.model_info.input_h[nlayer]
                            transfer_event_idx = self.fm_data_transfer_event_idx[nlayer-1][pos][pe_pos]
                            pre_event.add(transfer_event_idx)
                            if eri_event_idx not in self.Computation_order[transfer_event_idx].proceeding_event:
                                self.Computation_order[transfer_event_idx].proceeding_event.append(eri_event_idx)
                    eri_preceding_count = len(pre_event)
                    eri_inputs  = edram_read_data
                    eri_outputs = 0
                    event = EventMetaData("edram_rd", eri_position_idx, eri_preceding_count, [eri_event_idx+1], nlayer, eri_inputs, eri_outputs)
                    self.Computation_order.append(event)
                    ### input requirement
                    #
                   #---------------------#

                   #---Event: pooling---#
                    pooling_event_idx = len(self.Computation_order)
                    pooling_position_idx = (rty_idx, rtx_idx, pey_idx, pex_idx)
                    pooling_preceding_count = 1
                    pooling_amount = len(pe_inputs) # 做幾次
                    pooling_inputs  = pooling_amount
                    pooling_outputs = []
                    for pool_inp in pe_inputs:
                        data = pool_inp[0]
                        pooling_outputs.append(data)
                    event = EventMetaData("pooling", pooling_position_idx, pooling_preceding_count, [], nlayer, pooling_inputs, pooling_outputs)
                    self.Computation_order.append(event)
                   #--------------------#

                    des_pe_dict = dict() # {PE1: [data1, data2], PE2: [data1, data3]}
                    for d in pooling_outputs:
                        h = d[1]
                        w = d[2]
                        c = d[3]
                        pos = w + h * self.model_info.input_w[nlayer+1] + \
                                c * self.model_info.input_w[nlayer+1] * self.model_info.input_h[nlayer+1]
                        if self.model_info.layer_list[nlayer+1].layer_type != "fully":
                            data = (nlayer+1, h, w, c)
                        else:
                            data = (nlayer+1, pos, 0, 0)
                        des_pe_set = self.fm_data_used_pe_idx[nlayer+1][pos]
                        for des_pe in des_pe_set:
                            if des_pe in des_pe_dict:
                                des_pe_dict[des_pe].append(data)
                            else:
                                des_pe_dict[des_pe] = [data]
                    for des_pe in des_pe_dict: # 一個目的地PE生一個data transfer event
                       #---Event: data_transfer---#
                        transfer_event_idx = len(self.Computation_order)
                        data_transfer_src = (rty_idx, rtx_idx, pey_idx, pex_idx)
                        data_transfer_des = des_pe
                        transfer_position_idx = [data_transfer_src, data_transfer_des]
                        transfer_preceding_count = 1
                        transfer_inputs  = 0
                        transfer_outputs = des_pe_dict[des_pe]
                        event = EventMetaData("data_transfer", transfer_position_idx, transfer_preceding_count, [], nlayer, transfer_inputs, transfer_outputs)
                        self.Computation_order.append(event)
                        self.Computation_order[pooling_event_idx].proceeding_event.append(transfer_event_idx)
                        
                        # 先記錄data transfer event的index, 在生下一層的eDRAM read才能接dependency
                        for data in transfer_outputs:
                            if self.model_info.layer_list[nlayer+1].layer_type != "fully":
                                h = data[1]
                                w = data[2]
                                c = data[3]
                                pos = w + h * self.model_info.input_w[nlayer+1] + \
                                    c * self.model_info.input_w[nlayer+1] * self.model_info.input_h[nlayer+1]
                            else:
                                pos = data[1]
                            self.fm_data_transfer_event_idx[nlayer][pos][data_transfer_des] = transfer_event_idx
                       #--------------------------#
        print('Order generated!')

    def print_order(self):
        self.edram_rd_ir_ctr = 0
        self.cu_op_ctr  = 0
        self.pe_saa_ctr = 0
        self.activation_ctr = 0
        self.pooling_ctr = 0
        self.edram_wr_ctr = 0
        self.edram_rd_pool_ctr = 0
        self.data_transfer_ctr = 0
        for e in self.Computation_order:
            t = e.event_type
            if t == "edram_rd_ir":
                self.edram_rd_ir_ctr += 1
            elif t == "cu_operation":
                self.cu_op_ctr += 1
            elif t == "pe_saa":
                self.pe_saa_ctr += 1
            elif t == "activation":
                self.activation_ctr += 1
            elif t == "edram_wr":
                self.edram_wr_ctr += 1
            elif t == "edram_rd_pool":
                self.edram_rd_pool_ctr += 1
            elif t == "pooling":
                self.pooling_ctr += 1
            elif t == "data_transfer":
                self.data_transfer_ctr += 1
            else:
                print("event type error:", t)

        print("edram_rd_ir_ctr", self.edram_rd_ir_ctr)
        print("cu_op_ctr", self.cu_op_ctr)
        print("pe_saa_ctr", self.pe_saa_ctr)
        print("activation_ctr", self.activation_ctr)
        print("edram_wr_ctr", self.edram_wr_ctr)
        print("edram_rd_pool_ctr", self.edram_rd_pool_ctr)
        print("data_transfer_ctr", self.data_transfer_ctr)
        print("total", len(self.Computation_order))

        if True:
            for e in self.Computation_order:
                print(self.Computation_order.index(e), e)

    def old(self):
        pass
        # old version #
        '''
            if self.model_info.layer_list[nlayer].layer_type == "convolution":
                ### Event: edram_rd_ir
                for cu_pos in self.cu_traverse_idx:
                    pe_pos = cu_pos[:-2]
                    rty_idx, rtx_idx= cu_pos[0], cu_pos[1]
                    pey_idx, pex_idx = cu_pos[2], cu_pos[3]
                    cuy_idx, cux_idx = cu_pos[4], cu_pos[5]

                    # 算CU內的所有XB，最多需要多少組input
                    max_xb_input_len = 0
                    for xby_idx in range(HW().Xbar_num_y):
                        for xbx_idx in range(HW().Xbar_num_x):
                            xbar_inputs = self.mp_info.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx][xbx_idx][nlayer]
                            num_inp = len(xbar_inputs)
                            max_xb_input_len = max(max_xb_input_len, num_inp)

                    # 每一組input產生一個edram read event
                    for nInp in range(max_xb_input_len):
                        data_feed_to_cu = [] # 一次edram read的資料
                        for xby_idx in range(HW().Xbar_num_y):
                            for xbx_idx in range(HW().Xbar_num_x):
                                xbar_inputs = self.mp_info.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx][xbx_idx][nlayer]
                                if nInp < len(xbar_inputs):
                                    inp = xbar_inputs[nInp].inputs
                                else:
                                    inp = []
                                
                                # inp紀錄的位置為padding後的位置
                                if self.model_info.layer_list[nlayer].padding == "SAME":
                                    for d in inp:
                                        h = d[1] - self.model_info.pad[nlayer]
                                        w = d[2] - self.model_info.pad[nlayer]
                                        c = d[3]
                                        # 不紀錄pading的值
                                        if w >= 0 and w < self.model_info.input_w[nlayer] and h >= 0 and h < self.model_info.input_h[nlayer]:
                                            data = (nlayer, h, w, c)
                                            if data not in data_feed_to_cu:
                                                data_feed_to_cu.append(data)
                                else: # padding == "VALID"
                                    for d in inp:
                                        data = [nlayer] + d[1:]
                                        data = tuple(data)
                                        if data not in data_feed_to_cu:
                                            data_feed_to_cu.append(data)

                        eri_event_idx = len(self.Computation_order)
                        if nlayer == 0:
                            eri_preceding_count = 0
                        else:
                            eri_preceding_count = len(data_feed_to_cu)
                            # add dependency
                            for data in data_feed_to_cu:
                                pre_event_idx = self.transfer_mat[nlayer-1][data[1]][data[2]][data[3]][pe_pos]
                                self.Computation_order[pre_event_idx].proceeding_event.append(eri_event_idx)
                        eri_position_idx = cu_pos
                        eri_input_sequence  = data_feed_to_cu
                        eri_output_sequence = 0
                        event = EventMetaData("edram_rd_ir", eri_position_idx, eri_preceding_count, [eri_event_idx+1], nlayer, eri_input_sequence, eri_output_sequence)
                        self.Computation_order.append(event)

                        # input requirement
                        pe_id = cu_pos[3] + cu_pos[2]*HW().PE_num_x + \
                                cu_pos[1]*HW().PE_num + cu_pos[0]*HW().PE_num*HW().Router_num_x
                        for d in data_feed_to_cu:
                            pos = d[2] + d[1]*self.model_info.input_w[nlayer] + d[3]*self.model_info.input_w[nlayer]*self.model_info.input_h[nlayer] # w + h*width + c*height*width
                            self.free_buffer_controller.input_require[pe_id][nlayer][pos] += 1

                ### Event: cu_operation
                        cu_operation_idx = len(self.Computation_order)
                        num_ou_in_xb = dict()
                        max_ou = 0
                        for xby_idx in range(HW().Xbar_num_y):
                            for xbx_idx in range(HW().Xbar_num_x):
                                xbar_inputs  = self.mp_info.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx][xbx_idx][nlayer]
                                if nInp < len(xbar_inputs):
                                    inp = xbar_inputs[nInp]
                                else:
                                    continue

                                # Total ou in the event
                                num_ou_h = math.ceil(len(inp.inputs)/ HW().OU_h)
                                num_ou_w = math.ceil(inp.Cols  / HW().OU_w)
                                num_ou = num_ou_h * num_ou_w * self.model_info.input_bit

                                xb_idx = xbx_idx + xby_idx * HW().Xbar_num_x
                                num_ou_in_xb[xb_idx] = num_ou
                                max_ou = max(num_ou, max_ou)

                                num_input = inp.inputs[0][0]

                                filter_list = inp.Filters # 此xbar 會計算到的 filter
                                for nfilter in filter_list:
                                    grid = self.pe_saa_mat[nlayer][num_input][nfilter]
                                    if grid == 0.0:
                                        self.pe_saa_mat[nlayer][num_input][nfilter] = []
                                    if cu_operation_idx not in self.pe_saa_mat[nlayer][num_input][nfilter]:
                                        self.pe_saa_mat[nlayer][num_input][nfilter].append(cu_operation_idx)
                        
                        position_idx = cu_pos
                        preceding_count = 1
                        cu_op_inputs  = max_ou # 最多要幾次ou做完
                        cu_op_outputs = num_ou_in_xb
                        event = EventMetaData("cu_operation", position_idx, preceding_count, [], nlayer, cu_op_inputs, cu_op_outputs)
                        self.Computation_order.append(event)

                ### Event: edram_wr, data_transfer (for pe_saa), pe_saa
                windowlen_w = self.model_info.input_w[nlayer+1]
                windowlen_h = self.model_info.input_h[nlayer+1]
                for nfilter in range(self.model_info.filter_n[nlayer]):
                    for window_h in range(windowlen_h):
                        for window_w in range(windowlen_w):
                            num_input = window_h * windowlen_w + window_w
                            preceding_list = self.pe_saa_mat[nlayer][num_input][nfilter] 
                            first_pre_event_idx = preceding_list[0]  # do pe_saa in first pe of preceding cu_saa event
                            do_pe_saa_pos = self.Computation_order[first_pre_event_idx].position_idx[:-2]

                ### Event: edram_wr, data_transfer (for pe_saa)
                            pe_saa_precding_event_idx = []
                            preceding_pe = dict()
                            for pre_event_idx in preceding_list:
                                if self.Computation_order[pre_event_idx].position_idx[:-2] != do_pe_saa_pos: # in other pe
                                    if self.Computation_order[pre_event_idx].position_idx not in preceding_pe:
                                        preceding_pe[self.Computation_order[pre_event_idx].position_idx[:-2]] = [pre_event_idx]
                                    else:
                                        preceding_pe[self.Computation_order[pre_event_idx].position_idx[:-2]].append(pre_event_idx)
                                else: # in same PE
                                    pe_saa_precding_event_idx.append(pre_event_idx)
                            # other PEs
                            for pe_idx in preceding_pe:
                                # Event: pe_saa
                                pe_saa_event_idx = len(self.Computation_order)
                                pe_saa_pos = pe_idx
                                pe_saa_preceding_count = len(preceding_pe[pe_idx])
                                saa_amount = pe_saa_preceding_count
                                preceding_tmp_data = ()
                                pe_saa_inputs  = saa_amount
                                pe_saa_outputs = preceding_tmp_data
                                event = EventMetaData("pe_saa", pe_saa_pos, pe_saa_preceding_count, [pe_saa_event_idx+1], nlayer, pe_saa_inputs, pe_saa_outputs)
                                self.Computation_order.append(event)
                                # add dependency
                                for cu_op_event_idx in preceding_pe[pe_idx]:
                                    self.Computation_order[cu_op_event_idx].proceeding_event.append(pe_saa_event_idx)

                                # Event: edram_wr
                                edram_wr_event_idx = len(self.Computation_order)
                                do_edram_wr_pos = pe_idx
                                edram_wr_preceding_count = 1 # pe_saa
                                edram_wr_inputs  = 0
                                edram_wr_outputs = (nlayer, window_h, window_w, nfilter, "u")
                                event = EventMetaData("edram_wr", do_edram_wr_pos, edram_wr_preceding_count, [edram_wr_event_idx+1], nlayer, edram_wr_inputs, edram_wr_outputs)
                                self.Computation_order.append(event)

                                # Event: data_transfer
                                data_transfer_idx = len(self.Computation_order)
                                source_pe_idx      = pe_idx
                                destination_pe_idx = do_pe_saa_pos
                                data_transfer_preceding_count = 1
                                transfer_inputs  = 0
                                transfer_outputs = (nlayer, window_h, window_w, nfilter, "u")
                                event = EventMetaData("data_transfer", [source_pe_idx, destination_pe_idx], data_transfer_preceding_count, [], nlayer, transfer_inputs, transfer_outputs)
                                self.Computation_order.append(event)
                                pe_saa_precding_event_idx.append(data_transfer_idx)

                ### Event: pe_saa
                            pe_saa_event_idx = len(self.Computation_order)
                            pe_saa_pos = do_pe_saa_pos
                            pe_saa_preceding_count = len(pe_saa_precding_event_idx)
                            saa_amount = pe_saa_preceding_count
                            preceding_tmp_data = (nlayer, nfilter, 0, 0, "u")
                            pe_saa_inputs  = saa_amount
                            pe_saa_outputs = preceding_tmp_data
                            event = EventMetaData("pe_saa", pe_saa_pos, pe_saa_preceding_count, [pe_saa_event_idx+1], nlayer, pe_saa_inputs, pe_saa_outputs)
                            self.Computation_order.append(event)
                            # add dependency
                            for event_idx in pe_saa_precding_event_idx:
                                self.Computation_order[event_idx].proceeding_event.append(pe_saa_event_idx)
                
                ### Event: activation
                            act_event_idx = len(self.Computation_order)
                            do_act_pos = do_pe_saa_pos 
                            act_preceding_count = 1
                            act_inputs  = 0
                            act_outputs = 0
                            event = EventMetaData("activation", do_act_pos, act_preceding_count, [act_event_idx+1], nlayer, act_inputs, act_outputs)
                            self.Computation_order.append(event)

                ### Event: edram_wr, data_transfer(between layer)
                            edram_wr_event_idx = len(self.Computation_order)
                            do_edram_wr_pos = do_pe_saa_pos
                            edram_wr_preceding_count = 1
                            edram_wr_inputs  = 0
                            edram_wr_outputs = (nlayer+1, window_h, window_w, nfilter)
                            if nlayer+1 < self.model_info.layer_length:
                                if self.model_info.layer_list[nlayer+1].layer_type == "fully":
                                    seq = window_w + window_h * windowlen_w + nfilter * windowlen_w * windowlen_h
                                    edram_wr_outputs  = (nlayer+1, seq, 0, 0)
                            event = EventMetaData("edram_wr", do_edram_wr_pos, edram_wr_preceding_count, [], nlayer, edram_wr_inputs, edram_wr_outputs)
                            self.Computation_order.append(event)

                ### Event: data_transfer
                            if nlayer < self.model_info.layer_length - 1: # 最後一層不用生transfer
                                edram_wr_event = event
                                data = edram_wr_outputs
                                if self.model_info.layer_list[nlayer+1].layer_type != "fully":
                                    dependency_pe = self.transfer_mat[nlayer][data[1]][data[2]][data[3]]
                                else:
                                    dependency_pe = self.transfer_mat[nlayer][data[1]]
                                dependency_index = dict()
                                for pe_pos in dependency_pe:
                                    if pe_pos == do_edram_wr_pos:
                                        dependency_index[pe_pos] = edram_wr_event_idx
                                    else:
                                        data_transfer_event_idx = len(self.Computation_order)
                                        dependency_index[pe_pos] = data_transfer_event_idx

                                        data_transfer_source = do_edram_wr_pos
                                        data_transfer_destination = pe_pos
                                        data_transfer_preceding_count = 1
                                        data_transfer_inputs  = 0
                                        data_transfer_outputs = edram_wr_event.outputs
                                        event = EventMetaData("data_transfer", [data_transfer_source, data_transfer_destination], data_transfer_preceding_count, [], nlayer, data_transfer_inputs, data_transfer_outputs)
                                        self.Computation_order.append(event)

                                        # add dependency
                                        edram_wr_event.proceeding_event.append(data_transfer_event_idx)
                                
                                        # input requirement
                                        pe_id = data_transfer_source[3] + data_transfer_source[2]*HW().PE_num_x + \
                                                data_transfer_source[1]*HW().PE_num + data_transfer_source[0]*HW().PE_num*HW().Router_num_x
                                        if self.model_info.layer_list[nlayer+1].layer_type != "fully":
                                            pos = data[2] + data[1]*self.model_info.input_w[nlayer+1] + data[3]*self.model_info.input_w[nlayer+1]*self.model_info.input_h[nlayer+1] # w + h*width + c*height*width
                                            self.free_buffer_controller.input_require[pe_id][nlayer+1][pos] += 1
                                        else:
                                            self.free_buffer_controller.input_require[pe_id][nlayer+1][data[1]] += 1
                                
                                if self.model_info.layer_list[nlayer+1].layer_type != "fully":
                                    self.transfer_mat[nlayer][data[1]][data[2]][data[3]] = dependency_index
                                else:
                                    self.transfer_mat[nlayer][data[1]] = dependency_index

            elif self.model_info.layer_list[nlayer].layer_type == "fully":
                ### Event: edram_rd_ir
                for cu_pos in self.cu_traverse_idx:
                    pe_pos = cu_pos[:-2]
                    rty_idx, rtx_idx = cu_pos[0], cu_pos[1]
                    pey_idx, pex_idx = cu_pos[2], cu_pos[3]
                    cuy_idx, cux_idx = cu_pos[4], cu_pos[5]

                    max_xb_input_len = 0
                    for xby_idx in range(HW().Xbar_num_y):
                        for xbx_idx in range(HW().Xbar_num_x):
                            xbar_inputs = self.mp_info.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx][xbx_idx][nlayer]
                            num_inp = len(xbar_inputs)
                            max_xb_input_len = max(max_xb_input_len, num_inp)

                    for nInp in range(max_xb_input_len):
                        data_feed_to_cu = []
                        for xby_idx in range(HW().Xbar_num_y):
                            for xbx_idx in range(HW().Xbar_num_x):
                                xbar_inputs = self.mp_info.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx][xbx_idx][nlayer]
                                if nInp < len(xbar_inputs):
                                    inp = xbar_inputs[nInp].inputs
                                else:
                                    inp = []
                                for d in inp:
                                    data = [nlayer] + d[1:]
                                    data = tuple(data)
                                    if data not in data_feed_to_cu:
                                        data_feed_to_cu.append(data)

                        eri_event_idx = len(self.Computation_order)
                        if nlayer == 0:
                            eri_preceding_count = 0
                        else:
                            eri_preceding_count = len(data_feed_to_cu)
                            # add dependency
                            for data in data_feed_to_cu:
                                pre_event_idx = self.transfer_mat[nlayer-1][data[1]][pe_pos]
                                self.Computation_order[pre_event_idx].proceeding_event.append(eri_event_idx)
                        eri_position_idx = cu_pos
                        eri_input_sequence  = data_feed_to_cu
                        eri_output_sequence = 0
                        event = EventMetaData("edram_rd_ir", eri_position_idx, eri_preceding_count, [eri_event_idx+1], nlayer, eri_input_sequence, eri_output_sequence)
                        self.Computation_order.append(event)

                        # input requirement
                        pe_id = cu_pos[3] + cu_pos[2]*HW().PE_num_x + \
                                cu_pos[1]*HW().PE_num + cu_pos[0]*HW().PE_num*HW().Router_num_x
                        for d in data_feed_to_cu:
                            pos = d[1]
                            self.free_buffer_controller.input_require[pe_id][nlayer][pos] += 1

                ### Event: cu_operation
                        cu_operation_idx = len(self.Computation_order)
                        num_ou_in_xb = dict()
                        max_ou = 0
                        for xby_idx in range(HW().Xbar_num_y):
                            for xbx_idx in range(HW().Xbar_num_x):
                                xbar_inputs  = self.mp_info.layer_mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cuy_idx][cux_idx][xby_idx][xbx_idx][nlayer]
                                if nInp < len(xbar_inputs):
                                    inp = xbar_inputs[nInp]
                                else:
                                    continue
                                    
                                # Total ou in the event
                                num_ou_h = math.ceil(len(inp.inputs)/ HW().OU_h)
                                num_ou_w = math.ceil(inp.Cols  / HW().OU_w)
                                num_ou = num_ou_h * num_ou_w * self.model_info.input_bit

                                xb_idx = xbx_idx + xby_idx * HW().Xbar_num_x
                                num_ou_in_xb[xb_idx] = num_ou # Q1: xb_idx -> (0,0)
                                max_ou = max(num_ou, max_ou)

                                num_input = 0 #num_input = inp.inputs[0][0]

                                filter_list = inp.Filters # 此xbar 會計算到的 filter
                                for nfilter in filter_list:
                                    grid = self.pe_saa_mat[nlayer][num_input][nfilter]
                                    if grid == 0.0: # 優: init直接設定成list
                                        self.pe_saa_mat[nlayer][num_input][nfilter] = []
                                    if cu_operation_idx not in self.pe_saa_mat[nlayer][num_input][nfilter]:
                                        self.pe_saa_mat[nlayer][num_input][nfilter].append(cu_operation_idx)
                        
                        position_idx = cu_pos
                        preceding_count = 1
                        cu_op_inputs  = max_ou # 最多要幾次ou做完
                        cu_op_outputs = num_ou_in_xb
                        event = EventMetaData("cu_operation", position_idx, preceding_count, [], nlayer, cu_op_inputs, cu_op_outputs)
                        self.Computation_order.append(event)

                ### Event: pe_saa, edram_wr, data_transfer
                for nfilter in range(self.model_info.filter_n[nlayer]):
                    num_input = 0 # fully只有一組input vector
                    preceding_list = self.pe_saa_mat[nlayer][num_input][nfilter]
                    first_pre_event_idx = preceding_list[0]  # do pe_saa in first pe of preceding cu_saa event
                    do_pe_saa_pos = self.Computation_order[first_pre_event_idx].position_idx[:-2]

                ### Event: pe_saa, edram_wr, data_transfer
                    pe_saa_precding_event_idx = []
                    preceding_pe = dict()
                    for pre_event_idx in preceding_list:
                        if self.Computation_order[pre_event_idx].position_idx[:-2] != do_pe_saa_pos: # in other PE
                            if self.Computation_order[pre_event_idx].position_idx[:-2] not in preceding_pe:
                                preceding_pe[self.Computation_order[pre_event_idx].position_idx[:-2]] = [pre_event_idx]
                            else:
                                preceding_pe[self.Computation_order[pre_event_idx].position_idx[:-2]].append(pre_event_idx)
                        else: # in same PE
                            pe_saa_precding_event_idx.append(pre_event_idx)
                    # other PEs
                    for pe_idx in preceding_pe:
                        # Event: pe_saa
                        pe_saa_event_idx = len(self.Computation_order)
                        pe_saa_pos = pe_idx
                        pe_saa_preceding_count = len(preceding_pe[pe_idx])
                        saa_amount = pe_saa_preceding_count
                        preceding_tmp_data = ()
                        pe_saa_inputs  = saa_amount
                        pe_saa_outputs = preceding_tmp_data
                        event = EventMetaData("pe_saa", pe_saa_pos, pe_saa_preceding_count, [pe_saa_event_idx+1], nlayer, pe_saa_inputs, pe_saa_outputs)
                        self.Computation_order.append(event)
                        # add dependency
                        for cu_op_event_idx in preceding_pe[pe_idx]:
                            self.Computation_order[cu_op_event_idx].proceeding_event.append(pe_saa_event_idx)

                        # Event: edram_wr
                        edram_wr_event_idx = len(self.Computation_order)
                        do_edram_wr_pos = pe_idx
                        edram_wr_preceding_count = 1 # pe_saa
                        edram_wr_inputs  = 0
                        edram_wr_outputs = (nlayer, nfilter, 0, 0, "u") # Q2: 不同pe應該寫不同的質
                        event = EventMetaData("edram_wr", do_edram_wr_pos, edram_wr_preceding_count, [edram_wr_event_idx+1], nlayer, edram_wr_inputs, edram_wr_outputs)
                        self.Computation_order.append(event)

                        # Event: data_transfer
                        data_transfer_idx = len(self.Computation_order)
                        source_pe_idx      = pe_idx
                        destination_pe_idx = do_pe_saa_pos
                        data_transfer_preceding_count = 1
                        transfer_inputs  = 0
                        transfer_outputs = (nlayer, nfilter, 0, 0, "u") # Q3: 不同pe應該寫不同的質
                        event = EventMetaData("data_transfer", [source_pe_idx, destination_pe_idx], data_transfer_preceding_count, [], nlayer, transfer_inputs, transfer_outputs)
                        self.Computation_order.append(event)
                        pe_saa_precding_event_idx.append(data_transfer_idx)

                ### Event: pe_saa
                    pe_saa_event_idx = len(self.Computation_order)
                    pe_saa_pos = do_pe_saa_pos
                    pe_saa_preceding_count = len(pe_saa_precding_event_idx)
                    saa_amount = pe_saa_preceding_count
                    preceding_tmp_data = (nlayer, nfilter, 0, 0, "u") # Q4: 做完pe_saa要free掉的資料, 不同pe應該寫不同的質
                    pe_saa_inputs  = saa_amount
                    pe_saa_outputs = preceding_tmp_data
                    event = EventMetaData("pe_saa", pe_saa_pos, pe_saa_preceding_count, [pe_saa_event_idx+1], nlayer, pe_saa_inputs, pe_saa_outputs)
                    self.Computation_order.append(event)
                    # add dependency
                    for event_idx in pe_saa_precding_event_idx:
                        self.Computation_order[event_idx].proceeding_event.append(pe_saa_event_idx)

                ### Event: activation
                    act_event_idx = len(self.Computation_order)
                    do_act_pos = do_pe_saa_pos
                    act_preceding_count = 1
                    act_inputs  = 0
                    act_outputs = 0
                    event = EventMetaData("activation", do_act_pos, act_preceding_count, [act_event_idx+1], nlayer, act_inputs, act_outputs)
                    self.Computation_order.append(event)

                ### Event: edram_wr, data_transfer(between layer)
                    edram_wr_event_idx = len(self.Computation_order)
                    do_edram_wr_pos = do_pe_saa_pos
                    edram_wr_preceding_count = 1
                    edram_wr_inputs  = 0
                    edram_wr_outputs = (nlayer+1, nfilter, 0, 0)
                    event = EventMetaData("edram_wr", do_edram_wr_pos, edram_wr_preceding_count, [], nlayer, edram_wr_inputs, edram_wr_outputs)
                    self.Computation_order.append(event) 
               
                ### Event: data_transfer
                    if nlayer < self.model_info.layer_length - 1: # 最後一層不用生transfer
                        edram_wr_event = event
                        dependency_pe = self.transfer_mat[nlayer][nfilter] # set
                        dependency_index = dict()
                        for pe_pos in dependency_pe: # pe destination
                            if pe_pos == do_edram_wr_pos:
                                dependency_index[pe_pos] = edram_wr_event_idx
                            else:
                                data_transfer_event_idx = len(self.Computation_order)
                                dependency_index[pe_pos] = data_transfer_event_idx

                                data_transfer_source = do_edram_wr_pos
                                data_transfer_destination = pe_pos
                                data_transfer_preceding_count = 1
                                data_transfer_inputs  = 0
                                data_transfer_outputs = (nlayer+1, nfilter, 0, 0)
                                event = EventMetaData("data_transfer", [data_transfer_source, data_transfer_destination], data_transfer_preceding_count, [], nlayer, data_transfer_inputs, data_transfer_outputs)
                                self.Computation_order.append(event)

                                # add dependency
                                edram_wr_event.proceeding_event.append(data_transfer_event_idx)
                                
                                # input requirement
                                pe_id = data_transfer_source[3] + data_transfer_source[2]*HW().PE_num_x + \
                                        data_transfer_source[1]*HW().PE_num + data_transfer_source[0]*HW().PE_num*HW().Router_num_x
                                self.free_buffer_controller.input_require[pe_id][nlayer+1][nfilter] += 1

                        self.transfer_mat[nlayer][nfilter] = dependency_index # set換成dict

            elif self.model_info.layer_list[nlayer].layer_type == "pooling":
                for rty_idx in range(HW().Router_num_y):
                    for rtx_idx in range(HW().Router_num_x):
                        for pey_idx in range(HW().PE_num_y):
                            for pex_idx in range(HW().PE_num_x):
                                pe_pos = (rty_idx, rtx_idx, pey_idx, pex_idx)
                                for Inputs in self.mp_info.layer_mapping_to_pe[rty_idx][rtx_idx][pey_idx][pex_idx][nlayer]:
                ### Event: edram_rd_pool
                                    for inputs in Inputs:
                                        pool_inputs = []
                                        pool_event_idx = len(self.Computation_order)
                                        pool_preceding_count = len(inputs)
                                        # add dependency
                                        for data in inputs:
                                            pool_inputs.append(tuple(data[1:]))
                                            data = data[2:] # [h, w, c]
                                            pre_event_idx = self.transfer_mat[nlayer-1][data[0]][data[1]][data[2]][pe_pos]
                                            self.Computation_order[pre_event_idx].proceeding_event.append(pool_event_idx)

                                        pool_position_idx = pe_pos
                                        pool_inputs  = pool_inputs
                                        pool_outputs = 0
                                        event = EventMetaData("edram_rd_pool", pool_position_idx, pool_preceding_count, [pool_event_idx+1], nlayer, pool_inputs, pool_outputs)
                                        self.Computation_order.append(event)

                                        # input require
                                        pe_id = pe_pos[3] + pe_pos[2]*HW().PE_num_x + \
                                                pe_pos[1]*HW().PE_num + pe_pos[0]*HW().PE_num*HW().Router_num_x
                                        for data in pool_inputs:
                                            d = data[1:]
                                            pos = d[1] + d[0]*self.model_info.input_w[nlayer] + d[2]*self.model_info.input_w[nlayer]*self.model_info.input_h[nlayer] # w + h*width + c*height*width
                                            self.free_buffer_controller.input_require[pe_id][nlayer][pos] += 1

                ### Event: edram_wr
                                        edram_wr_event_idx = len(self.Computation_order)
                                        do_edram_wr_pos = pe_pos
                                        edram_wr_preceding_count = 1
                                        edram_wr_inputs  = 0
                                        edram_wr_outputs = (nlayer+1,
                                                            inputs[0][2] // self.model_info.pooling_strides[nlayer], 
                                                            inputs[0][3] // self.model_info.pooling_strides[nlayer], 
                                                            inputs[0][4])

                                        if nlayer+1 < self.model_info.layer_length:
                                            if self.model_info.layer_list[nlayer+1].layer_type == "fully":
                                                seq = edram_wr_outputs[1] * self.model_info.input_w[nlayer+1] + edram_wr_outputs[2] + edram_wr_outputs[3] * self.model_info.input_h[nlayer+1] * self.model_info.input_w[nlayer+1]
                                                edram_wr_outputs  = (nlayer+1, seq, 0, 0)
                                        event = EventMetaData("edram_wr", do_edram_wr_pos, edram_wr_preceding_count, [], nlayer, edram_wr_inputs, edram_wr_outputs)
                                        self.Computation_order.append(event)

                ### Event: data_transfer
                                        if nlayer < self.model_info.layer_length - 1: # 最後一層不用生transfer
                                            edram_wr_event = event
                                            data = edram_wr_outputs
                                            if self.model_info.layer_list[nlayer+1].layer_type != "fully":
                                                dependency_pe = self.transfer_mat[nlayer][data[1]][data[2]][data[3]]
                                            else:
                                                dependency_pe = self.transfer_mat[nlayer][data[1]]
                                            dependency_index = dict()
                                            for des_pe_pos in dependency_pe: # pe destination
                                                if des_pe_pos == do_edram_wr_pos:
                                                    dependency_index[des_pe_pos] = edram_wr_event_idx
                                                else:
                                                    data_transfer_event_idx = len(self.Computation_order)
                                                    dependency_index[des_pe_pos] = data_transfer_event_idx

                                                    data_transfer_source = do_edram_wr_pos
                                                    data_transfer_destination = des_pe_pos
                                                    data_transfer_preceding_count = 1
                                                    data_transfer_inputs  = 0
                                                    data_transfer_outputs = edram_wr_event.outputs
                                                    event = EventMetaData("data_transfer", [data_transfer_source, data_transfer_destination], data_transfer_preceding_count, [], nlayer, data_transfer_inputs, data_transfer_outputs)
                                                    self.Computation_order.append(event)
                                                    
                                                    # add dependency
                                                    edram_wr_event.proceeding_event.append(data_transfer_event_idx)
                                
                                                    # input requirement
                                                    pe_id = data_transfer_source[3] + data_transfer_source[2]*HW().PE_num_x + \
                                                            data_transfer_source[1]*HW().PE_num + data_transfer_source[0]*HW().PE_num*HW().Router_num_x
                                                    if self.model_info.layer_list[nlayer+1].layer_type != "fully":
                                                        pos = data[2] + data[1]*self.model_info.input_w[nlayer+1] + data[3]*self.model_info.input_w[nlayer+1]*self.model_info.input_h[nlayer+1] # w + h*width + c*height*width
                                                        self.free_buffer_controller.input_require[pe_id][nlayer+1][pos] += 1
                                                    else:
                                                        self.free_buffer_controller.input_require[pe_id][nlayer+1][data[1]] += 1
                                            if self.model_info.layer_list[nlayer+1].layer_type != "fully":
                                                self.transfer_mat[nlayer][data[1]][data[2]][data[3]] = dependency_index
                                            else:
                                                self.transfer_mat[nlayer][data[1]] = dependency_index
        '''
    def __str__(self):
        return str(self.__dict__)
