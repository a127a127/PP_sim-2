from EventMetaData import EventMetaData
from math import ceil, floor 

class OrderGenerator(object):
    def __init__(self, model_info, hw_config, mapping_information, trace):
        self.model_info = model_info
        self.hw_config = hw_config
        self.mp_info = mapping_information
        self.trace = trace
        #self.free_buffer_controller = FreeBufferController()

       #---紀錄每一筆feature map data會被哪些PE用到, 方便後面生data transfer---#
        self.fm_data_used_pe_idx = []
        for i in range(self.model_info.layer_length+1):
            arr = []
            for h in range(self.model_info.input_h[i] * self.model_info.input_w[i] * self.model_info.input_c[i]):
                arr.append(set())
            self.fm_data_used_pe_idx.append(arr)
        
        for nlayer in range(self.model_info.layer_length):
            layer_type = self.model_info.layer_list[nlayer].layer_type
            if layer_type == "convolution":
                for pe_pos in self.mp_info.layer_used_component[nlayer]:
                    rty_idx, rtx_idx = pe_pos[0], pe_pos[1]
                    pey_idx, pex_idx = pe_pos[2], pe_pos[3]
                    cu_idx_set = self.mp_info.layer_used_component[nlayer][pe_pos]
                    for cu_idx in cu_idx_set:
                        for xb_idx in range(self.hw_config.Xbar_num):
                            for window_h in range(self.model_info.input_h[nlayer+1]):
                                for window_w in range(self.model_info.input_w[nlayer+1]):
                                    if (window_h,window_w) in self.mp_info.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx][nlayer]:
                                        xbar_mapping_info_list = self.mp_info.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx][nlayer][(window_h,window_w)]
                                        for xbar_mapping_info in xbar_mapping_info_list:
                                            inp_vector = xbar_mapping_info.inputs
                                            for data in inp_vector:
                                                if data != 0:
                                                    h,w,c = data[1], data[2], data[3]
                                                    data_pos = w + h * self.model_info.input_w[nlayer] + c * self.model_info.input_w[nlayer] * self.model_info.input_h[nlayer]
                                                    self.fm_data_used_pe_idx[nlayer][data_pos].add(pe_pos)
            elif layer_type == "fully":
                for pe_pos in self.mp_info.layer_used_component[nlayer]:
                    rty_idx, rtx_idx = pe_pos[0], pe_pos[1]
                    pey_idx, pex_idx = pe_pos[2], pe_pos[3]
                    cu_idx_set = self.mp_info.layer_used_component[nlayer][pe_pos]
                    for cu_idx in cu_idx_set:
                        for xb_idx in range(self.hw_config.Xbar_num):
                            if 0 in self.mp_info.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx][nlayer]:
                                xbar_mapping_info_list = self.mp_info.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx][nlayer][0]
                                for xbar_mapping_info in xbar_mapping_info_list:
                                    inp_vector = xbar_mapping_info.inputs
                                    for data in inp_vector:
                                        data_pos = data[1]
                                        self.fm_data_used_pe_idx[nlayer][data_pos].add(pe_pos)
            elif layer_type == "pooling":
                for pe_pos in self.mp_info.layer_used_component[nlayer]:
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
       
       #---後面用來紀錄feature map data transfer event的index---#
        self.fm_data_transfer_event_idx = []
        for nlayer in range(self.model_info.layer_length):
            self.fm_data_transfer_event_idx.append([])
            for i in range(self.model_info.input_h[nlayer+1]*self.model_info.input_w[nlayer+1]*self.model_info.input_c[nlayer+1]):
                self.fm_data_transfer_event_idx[nlayer].append(dict()) # {PE: transfer_event_idx}
       #-----------------------------------------------------#  

        self.transfer_feature_map_data_num  = 0
        self.transfer_intermediate_data_num = 0

       #---用來紀錄每個CU input window depenedency的event index---#
        self.CU_dependency_idx = []
        for rty_idx in range(self.hw_config.Router_num_y):
            for rtx_idx in range(self.hw_config.Router_num_x):
                for pe_idx in range(self.hw_config.PE_num):
                    for cu_idx in range(self.hw_config.CU_num):
                        self.CU_dependency_idx.append(-1)
       #-------------------------------------------------------# 
       #--用來紀錄每個PE內eDRAM的event index-----#
        self.PE_EDRAM_DEPENDENCY = True
        if self.PE_EDRAM_DEPENDENCY:
            self.PE_dependency_idx = []
            for rty_idx in range(self.hw_config.Router_num_y):
                for rtx_idx in range(self.hw_config.Router_num_x):
                    for pe_idx in range(self.hw_config.PE_num):
                        self.PE_dependency_idx.append(-1)
       #-------------------------------------#

        self.Computation_order = []
        self.generate_order()
        
        self.print_order()
       
    def generate_order(self):
        for nlayer in range(self.model_info.layer_length):
            layer_type = self.model_info.layer_list[nlayer].layer_type
            print("Generate layer", nlayer, layer_type)
            if layer_type == "convolution":
               #----決定每個filter的aggregator PE----#
                pe_filter_processing = dict() # {PE1:{"act":[f1, f2], "transfer": {des_pe1:[f3, f4], des_pe2:[f5,f6]}, "aggregate":[f7]}, PE2: ...}
                                              #   PE1做act的filter(f1&f2),  f3&f4會傳到PE1做arregate f5&f6傳到PE2,       f7在此PE aggregate
                # 每個filter在哪一些PEs算
                pe_operate_filter    = dict() # {PE1: {filter1, filter2, ...}, PE2: {filter2, filter3, ...}
                for pe_pos in self.mp_info.layer_used_component[nlayer]:
                    rty_idx, rtx_idx = pe_pos[0], pe_pos[1]
                    pey_idx, pex_idx = pe_pos[2], pe_pos[3]
                    cu_idx_set = self.mp_info.layer_used_component[nlayer][pe_pos]
                    operate_filter = set() # 此pe有算到哪些filter
                    for cu_idx in cu_idx_set:
                        for xb_idx in range(self.hw_config.Xbar_num):
                            if (0, 0) in self.mp_info.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx][nlayer]:
                                xbar_mapping_info_list = self.mp_info.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx][nlayer][(0,0)] # 以第一個window為主
                                for xbar_mapping_info in xbar_mapping_info_list:
                                    filter_list = xbar_mapping_info.Filters # 此xbar 會計算到的 filter
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
                                aggregator_dict["aggregator"] = pe_pos # 隨機指定一個PE當aggregator
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
               #-----------------------------------#

                for window in self.mp_info.window_order[nlayer]:
                    window_h, window_w = window[0], window[1]
                    
                    # 兩個用來記錄event index的字典 # for dependency
                    pe_saa_event_dict = dict()
                    wr_and_transfer_event_dict = dict()
                    for pe_pos in self.mp_info.layer_used_component[nlayer]:
                        pe_saa_event_dict[pe_pos] = []
                        wr_and_transfer_event_dict[pe_pos] = []

                   #---generate events--#
                    for pe_pos in self.mp_info.layer_used_component[nlayer]:
                        rty_idx, rtx_idx = pe_pos[0], pe_pos[1]
                        pey_idx, pex_idx = pe_pos[2], pe_pos[3]
                        cu_idx_set = self.mp_info.layer_used_component[nlayer][pe_pos]
                        for cu_idx in cu_idx_set:
                           #---Event: edram_rd_ir---#
                            eri_event_idx = len(self.Computation_order)
                            eri_position_idx = (rty_idx, rtx_idx, pey_idx, pex_idx, cu_idx)
                            
                            # 準備edram read data
                            edram_read_data = set()
                            for xb_idx in range(self.hw_config.Xbar_num):
                                if (window_h,window_w) in self.mp_info.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx][nlayer]:
                                    xbar_mapping_info_list = self.mp_info.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx][nlayer][(window_h,window_w)]
                                    for xbar_mapping_info in xbar_mapping_info_list:
                                        inp_vector = xbar_mapping_info.inputs
                                        for data in inp_vector:
                                            if data != 0:
                                                edram_read_data.add(data)
                            edram_read_data = list(edram_read_data)
                            
                            if not edram_read_data:
                                # 某些特定case下
                                # 此CU沒有任何input vector要做 有可能剛好是padding 0
                                # 後面會生出edram wr要修正
                                print("Generate Order Error: no edram read data")
                                exit()
                                #continue

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
                            

                            # dependency: different windows dependency
                            cu_id = cu_idx + pex_idx * self.hw_config.CU_num + pey_idx * self.hw_config.PE_num_x * self.hw_config.CU_num + \
                                    rtx_idx * self.hw_config.PE_num * self.hw_config.CU_num + rty_idx * self.hw_config.Router_num_x * self.hw_config.PE_num * self.hw_config.CU_num
                            pre_cu_op_id = self.CU_dependency_idx[cu_id]
                            if pre_cu_op_id > -1:
                                eri_preceding_count += 1
                                self.Computation_order[pre_cu_op_id].proceeding_event.append(eri_event_idx)
                            
                            # dependency: eDRAM event dependency in same PE
                            if self.PE_EDRAM_DEPENDENCY:
                                pe_id =  pex_idx + pey_idx * self.hw_config.PE_num_x + \
                                    rtx_idx * self.hw_config.PE_num + rty_idx * self.hw_config.Router_num_x * self.hw_config.PE_num
                                pre_edram_id = self.PE_dependency_idx[pe_id]
                                if pre_edram_id != -1:
                                    eri_preceding_count += 1
                                    self.Computation_order[pre_edram_id].proceeding_event.append(eri_event_idx)
                                self.PE_dependency_idx[pe_id] = eri_event_idx

                            eri_inputs  = edram_read_data
                            eri_outputs = 0
                            event = EventMetaData("edram_rd_ir", eri_position_idx, eri_preceding_count, [eri_event_idx+1], nlayer, eri_inputs, eri_outputs)
                            self.Computation_order.append(event)
                            
                            ### input requirement
                            # pe_id = pex_idx + pey_idx*self.hw_config.PE_num_x + \
                            #         rtx_idx*self.hw_config.PE_num + rty_idx*self.hw_config.PE_num*self.hw_config.Router_num_x
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
                            for xb_idx in range(self.hw_config.Xbar_num):
                                if (window_h,window_w) in self.mp_info.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx][nlayer]:
                                    xbar_mapping_info_list = self.mp_info.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx][nlayer][(window_h,window_w)]
                                    for xbar_mapping_info in xbar_mapping_info_list:
                                        inp_vector = xbar_mapping_info.inputs
                                        num_ou_h = ceil(len(inp_vector)/ self.hw_config.OU_h)
                                        num_ou_w = ceil(xbar_mapping_info.Cols / self.hw_config.OU_w)
                                        num_ou = num_ou_h * num_ou_w * self.model_info.input_bit
                                        num_ou_in_xb[xb_idx] = num_ou
                                        max_ou = max(num_ou, max_ou)

                            cu_op_inputs  = [max_ou, num_ou_in_xb]
                            cu_op_outputs = 0

                            event = EventMetaData("cu_operation", cu_op_position_idx, preceding_count, [cu_operation_event_idx+1], nlayer, cu_op_inputs, cu_op_outputs)
                            self.Computation_order.append(event)
                            
                            self.CU_dependency_idx[cu_id] = cu_operation_event_idx
                           #-------------------------#

                           #---Event: pe_saa---#
                            pe_saa_event_idx = len(self.Computation_order)
                            pe_saa_position_idx = (rty_idx, rtx_idx, pey_idx, pex_idx)
                            preceding_count = 1

                            # Shift and add 要做多少次
                            cu_operate_filter = set()
                            for xb_idx in range(self.hw_config.Xbar_num):
                                if (window_h,window_w) in self.mp_info.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx][nlayer]:
                                    xbar_mapping_info_list = self.mp_info.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx][nlayer][(window_h,window_w)]
                                    for xbar_mapping_info in xbar_mapping_info_list:
                                        filter_list = xbar_mapping_info.Filters # 此xbar 會計算到的 filter
                                        for f in filter_list:
                                            cu_operate_filter.add(f)
                            saa_amount = len(cu_operate_filter)
                            pe_saa_inputs  = saa_amount

                            pe_saa_outputs = 0
                            event = EventMetaData("pe_saa", pe_saa_position_idx, preceding_count, [], nlayer, pe_saa_inputs, pe_saa_outputs)
                            self.Computation_order.append(event)
                            pe_saa_event_dict[pe_pos].append(pe_saa_event_idx)
                           #-------------------#
                        
                    for pe_pos in self.mp_info.layer_used_component[nlayer]:
                        rty_idx, rtx_idx = pe_pos[0], pe_pos[1]
                        pey_idx, pex_idx = pe_pos[2], pe_pos[3]
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

                                # dependency: eDRAM event dependency in same PE
                                if self.PE_EDRAM_DEPENDENCY:
                                    pe_id =  pex_idx + pey_idx * self.hw_config.PE_num_x + \
                                        rtx_idx * self.hw_config.PE_num + rty_idx * self.hw_config.Router_num_x * self.hw_config.PE_num
                                    pre_edram_id = self.PE_dependency_idx[pe_id]
                                    if pre_edram_id != -1:
                                        wr_preceding_count += 1
                                        self.Computation_order[pre_edram_id].proceeding_event.append(wr_event_idx)
                                    self.PE_dependency_idx[pe_id] = wr_event_idx
                            
                                event = EventMetaData("edram_wr", wr_position_idx, wr_preceding_count, [], nlayer, wr_inputs, wr_outputs)
                                self.Computation_order.append(event)
                                self.Computation_order[act_event_idx].proceeding_event.append(wr_event_idx) # dependency
                               #---------------------#

                            else: # 不是最後一層要生data transfer event傳到下一層所在PE
                                des_pe_dict = dict() # {PE1: [data1, data2], PE2: [data1, data3]
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
                                    
                                    self.transfer_feature_map_data_num += len(transfer_outputs)
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
                                
                                self.transfer_intermediate_data_num += len(transfer_outputs)
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

                            # dependency: eDRAM event dependency in same PE
                            if self.PE_EDRAM_DEPENDENCY:
                                pe_id =  pex_idx + pey_idx * self.hw_config.PE_num_x + \
                                    rtx_idx * self.hw_config.PE_num + rty_idx * self.hw_config.Router_num_x * self.hw_config.PE_num
                                pre_edram_id = self.PE_dependency_idx[pe_id]
                                if pre_edram_id != -1:
                                    wr_preceding_count += 1
                                    self.Computation_order[pre_edram_id].proceeding_event.append(wr_event_idx)
                                self.PE_dependency_idx[pe_id] = wr_event_idx

                            event = EventMetaData("edram_wr", wr_position_idx, wr_preceding_count, [], nlayer, wr_inputs, wr_outputs)
                            self.Computation_order.append(event)

                            for event_idx in pe_saa_event_dict[pe_pos]: # dependency
                                self.Computation_order[event_idx].proceeding_event.append(wr_event_idx)
                            wr_and_transfer_event_dict[pe_pos].append(wr_event_idx) # for dependency
                           #---------------------#
                    
                    for pe_pos in self.mp_info.layer_used_component[nlayer]:
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
                            
                            # dependency: eDRAM event dependency in same PE
                            if self.PE_EDRAM_DEPENDENCY:
                                pe_id =  pex_idx + pey_idx * self.hw_config.PE_num_x + \
                                    rtx_idx * self.hw_config.PE_num + rty_idx * self.hw_config.Router_num_x * self.hw_config.PE_num
                                pre_edram_id = self.PE_dependency_idx[pe_id]
                                if pre_edram_id != -1:
                                    eri_preceding_count += 1
                                    self.Computation_order[pre_edram_id].proceeding_event.append(eri_event_idx)
                                self.PE_dependency_idx[pe_id] = eri_event_idx

                            eri_inputs  = edram_read_data
                            eri_outputs = 0
                            event = EventMetaData("edram_rd", eri_position_idx, eri_preceding_count, [eri_event_idx+1], nlayer, eri_inputs, eri_outputs)
                            self.Computation_order.append(event)

                            for event_idx in wr_and_transfer_event_dict[pe_pos]:
                                self.Computation_order[event_idx].proceeding_event.append(eri_event_idx)
                            ### input requirement
                            #
                           #---------------------#

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

                                # dependency: eDRAM event dependency in same PE
                                if self.PE_EDRAM_DEPENDENCY:
                                    pe_id =  pex_idx + pey_idx * self.hw_config.PE_num_x + \
                                        rtx_idx * self.hw_config.PE_num + rty_idx * self.hw_config.Router_num_x * self.hw_config.PE_num
                                    pre_edram_id = self.PE_dependency_idx[pe_id]
                                    if pre_edram_id != -1:
                                        wr_preceding_count += 1
                                        self.Computation_order[pre_edram_id].proceeding_event.append(wr_event_idx)
                                    self.PE_dependency_idx[pe_id] = wr_event_idx

                                event = EventMetaData("edram_wr", wr_position_idx, wr_preceding_count, [], nlayer, wr_inputs, wr_outputs)
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
                                    
                                    self.transfer_feature_map_data_num += len(transfer_outputs)
                                   #--------------------------#     
                   #--------------------#
            
            elif layer_type == "fully":
               #----決定每個filter的aggregator PE----# # 和conv一樣
                pe_filter_processing = dict() # {PE1:{"act":[f1, f2], "transfer": {des_pe1:[f3, f4], des_pe2:[f5,f6]}, "aggregate":[f7]}, PE2: ...}

                # 每個filter在哪一些PEs算
                pe_operate_filter    = dict() # {PE1: {filter1, filter2, ...}, PE2: {filter2, filter3, ...}
                for pe_pos in self.mp_info.layer_used_component[nlayer]:
                    rty_idx, rtx_idx = pe_pos[0], pe_pos[1]
                    pey_idx, pex_idx = pe_pos[2], pe_pos[3]
                    cu_idx_set = self.mp_info.layer_used_component[nlayer][pe_pos]
                    operate_filter = set() # 此pe有算到哪些filter
                    for cu_idx in cu_idx_set:
                        for xb_idx in range(self.hw_config.Xbar_num):
                            if 0 in self.mp_info.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx][nlayer]:
                                xbar_mapping_info_list = self.mp_info.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx][nlayer][0]
                                for xbar_mapping_info in xbar_mapping_info_list:
                                    filter_list = xbar_mapping_info.Filters # 此xbar 會計算到的 filter
                                    for f in filter_list:
                                        operate_filter.add(f)
                    pe_operate_filter[pe_pos] = operate_filter
                
                # 每個filter的aggregator pe, 和non aggregator pe
                filter_aggregator = [] # {"aggregator": PE1, "non": [PE2, PE3]}
                for f in range(self.model_info.filter_n[nlayer]):
                    aggregator_dict = {"aggregator": 0, "non": []}
                    for pe_pos in pe_operate_filter:
                        if f in pe_operate_filter[pe_pos]:
                            if aggregator_dict["aggregator"] == 0:
                                aggregator_dict["aggregator"] = pe_pos # 隨機指定一個PE當aggregator
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
               #-----------------------------------#
                
               # 兩個用來記錄event index的字典 # for dependency
                pe_saa_event_dict = dict()
                wr_and_transfer_event_dict = dict()
                for pe_pos in self.mp_info.layer_used_component[nlayer]:
                    pe_saa_event_dict[pe_pos] = []
                    wr_and_transfer_event_dict[pe_pos] = []

               #---generate events---#
                for pe_pos in self.mp_info.layer_used_component[nlayer]:
                    rty_idx, rtx_idx = pe_pos[0], pe_pos[1]
                    pey_idx, pex_idx = pe_pos[2], pe_pos[3]
                    cu_idx_set = self.mp_info.layer_used_component[nlayer][pe_pos]
                    for cu_idx in cu_idx_set:
                       #---Event: edram_rd_ir---#
                        eri_event_idx = len(self.Computation_order)
                        eri_position_idx = (rty_idx, rtx_idx, pey_idx, pex_idx, cu_idx)
                                
                        # 準備edram read一次讀多少data
                        edram_read_data = set()
                        for xb_idx in range(self.hw_config.Xbar_num):
                            if 0 in self.mp_info.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx][nlayer]:
                                xbar_mapping_info_list = self.mp_info.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx][nlayer][0]
                                for xbar_mapping_info in xbar_mapping_info_list:
                                    inp_vector = xbar_mapping_info.inputs
                                    for data in inp_vector:
                                        if data != 0:
                                            edram_read_data.add(data)
                        edram_read_data = list(edram_read_data)

                        if not edram_read_data:
                            # 某些特定case下
                            # 此CU沒有任何input vector要做 有可能剛好是padding 0
                            # 後面會生出edram wr要修正
                            print("Generate Order Error: no edram read data")
                            exit()
                            #continue

                        # dependency: transfer -> edram_rd_ir
                        pre_event = set()
                        if nlayer != 0:
                            for data in edram_read_data:
                                pos = data[1] # h
                                transfer_event_idx = self.fm_data_transfer_event_idx[nlayer-1][pos][pe_pos]
                                pre_event.add(transfer_event_idx)
                                if eri_event_idx not in self.Computation_order[transfer_event_idx].proceeding_event:
                                    self.Computation_order[transfer_event_idx].proceeding_event.append(eri_event_idx)
                        
                        if nlayer == 0:
                            eri_preceding_count = 0
                        else:
                            eri_preceding_count = len(pre_event)

                        # dependency: different windows dependency
                        cu_id = cu_idx + pex_idx * self.hw_config.CU_num + pey_idx * self.hw_config.PE_num_x * self.hw_config.CU_num + \
                                rtx_idx * self.hw_config.PE_num * self.hw_config.CU_num + rty_idx * self.hw_config.Router_num_x * self.hw_config.PE_num * self.hw_config.CU_num
                        pre_cu_op_id = self.CU_dependency_idx[cu_id]
                        if pre_cu_op_id > -1:
                            eri_preceding_count += 1
                            self.Computation_order[pre_cu_op_id].proceeding_event.append(eri_event_idx)
                        
                        # dependency: eDRAM event dependency in same PE
                        if self.PE_EDRAM_DEPENDENCY:
                            pe_id =  pex_idx + pey_idx * self.hw_config.PE_num_x + \
                                rtx_idx * self.hw_config.PE_num + rty_idx * self.hw_config.Router_num_x * self.hw_config.PE_num
                            pre_edram_id = self.PE_dependency_idx[pe_id]
                            if pre_edram_id != -1:
                                eri_preceding_count += 1
                                self.Computation_order[pre_edram_id].proceeding_event.append(eri_event_idx)
                            self.PE_dependency_idx[pe_id] = eri_event_idx
                                
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
                        for xb_idx in range(self.hw_config.Xbar_num):
                            if 0 in self.mp_info.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx][nlayer]:
                                xbar_mapping_info_list = self.mp_info.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx][nlayer][0]
                                for xbar_mapping_info in xbar_mapping_info_list:
                                    inp_vector = xbar_mapping_info.inputs
                                    num_ou_h = ceil(len(inp_vector)/ self.hw_config.OU_h)
                                    num_ou_w = ceil(xbar_mapping_info.Cols / self.hw_config.OU_w)
                                    num_ou = num_ou_h * num_ou_w * self.model_info.input_bit
                                    num_ou_in_xb[xb_idx] = num_ou
                                    max_ou = max(num_ou, max_ou)

                        cu_op_inputs  = [max_ou, num_ou_in_xb]
                        cu_op_outputs = 0

                        event = EventMetaData("cu_operation", cu_op_position_idx, preceding_count, [cu_operation_event_idx+1], nlayer, cu_op_inputs, cu_op_outputs)
                        self.Computation_order.append(event)

                        self.CU_dependency_idx[cu_id] = cu_operation_event_idx
                       #-------------------------#

                       #---Event: pe_saa---#
                        pe_saa_event_idx = len(self.Computation_order)
                        pe_saa_position_idx = (rty_idx, rtx_idx, pey_idx, pex_idx)
                        preceding_count = 1

                        # Shift and add 要做多少次
                        cu_operate_filter = set()
                        for xb_idx in range(self.hw_config.Xbar_num):
                            if 0 in self.mp_info.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx][nlayer]:
                                xbar_mapping_info_list = self.mp_info.mapping_to_xbar[rty_idx][rtx_idx][pey_idx][pex_idx][cu_idx][xb_idx][nlayer][0]
                                for xbar_mapping_info in xbar_mapping_info_list:
                                    if xbar_mapping_info:
                                        filter_list = xbar_mapping_info.Filters # 此xbar 會計算到的 filter
                                        for f in filter_list:
                                            cu_operate_filter.add(f)
                        saa_amount = len(cu_operate_filter)
                        pe_saa_inputs  = saa_amount

                        pe_saa_outputs = 0
                        event = EventMetaData("pe_saa", pe_saa_position_idx, preceding_count, [], nlayer, pe_saa_inputs, pe_saa_outputs)
                        self.Computation_order.append(event)
                        pe_saa_event_dict[pe_pos].append(pe_saa_event_idx) # dependency: pe_saa -> act, write, transfer
                       #-------------------#

                for pe_pos in self.mp_info.layer_used_component[nlayer]:
                    rty_idx, rtx_idx = pe_pos[0], pe_pos[1]
                    pey_idx, pex_idx = pe_pos[2], pe_pos[3]
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

                            # dependency: eDRAM event dependency in same PE
                            if self.PE_EDRAM_DEPENDENCY:
                                pe_id =  pex_idx + pey_idx * self.hw_config.PE_num_x + \
                                    rtx_idx * self.hw_config.PE_num + rty_idx * self.hw_config.Router_num_x * self.hw_config.PE_num
                                pre_edram_id = self.PE_dependency_idx[pe_id]
                                if pre_edram_id != -1:
                                    wr_preceding_count += 1
                                    self.Computation_order[pre_edram_id].proceeding_event.append(wr_event_idx)
                                self.PE_dependency_idx[pe_id] = wr_event_idx
                        
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
                                
                                self.transfer_feature_map_data_num += len(transfer_outputs)
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
                            
                            self.transfer_intermediate_data_num += len(transfer_outputs)
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

                        # dependency: eDRAM event dependency in same PE
                        if self.PE_EDRAM_DEPENDENCY:
                            pe_id =  pex_idx + pey_idx * self.hw_config.PE_num_x + \
                                rtx_idx * self.hw_config.PE_num + rty_idx * self.hw_config.Router_num_x * self.hw_config.PE_num
                            pre_edram_id = self.PE_dependency_idx[pe_id]
                            if pre_edram_id != -1:
                                wr_preceding_count += 1
                                self.Computation_order[pre_edram_id].proceeding_event.append(wr_event_idx)
                            self.PE_dependency_idx[pe_id] = wr_event_idx

                        event = EventMetaData("edram_wr", wr_position_idx, wr_preceding_count, [], nlayer, wr_inputs, wr_outputs)
                        self.Computation_order.append(event)

                        for event_idx in pe_saa_event_dict[pe_pos]: # dependency
                            self.Computation_order[event_idx].proceeding_event.append(wr_event_idx)
                        wr_and_transfer_event_dict[pe_pos].append(wr_event_idx) # for dependency
                       #---------------------#

                for pe_pos in self.mp_info.layer_used_component[nlayer]:
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

                        # dependency: eDRAM event dependency in same PE
                        if self.PE_EDRAM_DEPENDENCY:
                            pe_id =  pex_idx + pey_idx * self.hw_config.PE_num_x + \
                                rtx_idx * self.hw_config.PE_num + rty_idx * self.hw_config.Router_num_x * self.hw_config.PE_num
                            pre_edram_id = self.PE_dependency_idx[pe_id]
                            if pre_edram_id != -1:
                                eri_preceding_count += 1
                                self.Computation_order[pre_edram_id].proceeding_event.append(eri_event_idx)
                            self.PE_dependency_idx[pe_id] = eri_event_idx
                        
                        eri_inputs  = edram_read_data
                        eri_outputs = 0
                        event = EventMetaData("edram_rd", eri_position_idx, eri_preceding_count, [eri_event_idx+1], nlayer, eri_inputs, eri_outputs)
                        self.Computation_order.append(event)

                        for event_idx in wr_and_transfer_event_dict[pe_pos]:
                            self.Computation_order[event_idx].proceeding_event.append(eri_event_idx)
                        ### input requirement
                        #
                       #---------------------#

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

                            # dependency: eDRAM event dependency in same PE
                            if self.PE_EDRAM_DEPENDENCY:
                                pe_id =  pex_idx + pey_idx * self.hw_config.PE_num_x + \
                                    rtx_idx * self.hw_config.PE_num + rty_idx * self.hw_config.Router_num_x * self.hw_config.PE_num
                                pre_edram_id = self.PE_dependency_idx[pe_id]
                                if pre_edram_id != -1:
                                    wr_preceding_count += 1
                                    self.Computation_order[pre_edram_id].proceeding_event.append(wr_event_idx)
                                self.PE_dependency_idx[pe_id] = wr_event_idx

                            event = EventMetaData("edram_wr", wr_position_idx, wr_preceding_count, [], nlayer, wr_inputs, wr_outputs)
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
                                
                                self.transfer_feature_map_data_num += len(transfer_outputs)
                               #--------------------------#
                                
               #=============#

            elif layer_type == "pooling":
                eDRAM_buffer_read_bits = self.hw_config.eDRAM_buffer_read_bits # per cycle
                eDRAM_read_data = floor(eDRAM_buffer_read_bits / self.model_info.input_bit) # per cycle
                pooling_data = self.model_info.pooling_h[nlayer] * self.model_info.pooling_w[nlayer] # per pooling
                num_pooling_per_cycle = floor(eDRAM_read_data / pooling_data) # 一個cycle最多可以做幾個pooling
                for pe_pos in self.mp_info.layer_used_component[nlayer]:
                    rty_idx, rtx_idx = pe_pos[0], pe_pos[1]
                    pey_idx, pex_idx = pe_pos[2], pe_pos[3]

                    pe_inputs = self.mp_info.mapping_to_pe[rty_idx][rtx_idx][pey_idx][pex_idx][nlayer]
                    total_num_pooling = len(pe_inputs)
                    num_pooling_events = ceil(total_num_pooling / num_pooling_per_cycle)
                    for num_pooling in range(num_pooling_events):
                        inputs = pe_inputs[num_pooling*num_pooling_per_cycle : (num_pooling+1)*num_pooling_per_cycle]
                       #---Event: edram_rd---#
                        eri_event_idx = len(self.Computation_order)
                        eri_position_idx = (rty_idx, rtx_idx, pey_idx, pex_idx)
                        edram_read_data = set()
                        for pool_inp in inputs:
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

                        # dependency: eDRAM event dependency in same PE
                        if self.PE_EDRAM_DEPENDENCY:
                            pe_id =  pex_idx + pey_idx * self.hw_config.PE_num_x + \
                                rtx_idx * self.hw_config.PE_num + rty_idx * self.hw_config.Router_num_x * self.hw_config.PE_num
                            pre_edram_id = self.PE_dependency_idx[pe_id]
                            if pre_edram_id != -1:
                                eri_preceding_count += 1
                                self.Computation_order[pre_edram_id].proceeding_event.append(eri_event_idx)
                            self.PE_dependency_idx[pe_id] = eri_event_idx

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
                        pooling_amount = len(inputs) # 做幾次
                        pooling_inputs  = pooling_amount
                        pooling_outputs = []
                        for pool_inp in inputs:
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
                            
                            self.transfer_feature_map_data_num += len(transfer_outputs)
                           #--------------------------#
        
        print('Order generated!')

    def print_order(self):
        self.edram_rd_ir_ctr = 0
        self.cu_op_ctr  = 0
        self.pe_saa_ctr = 0
        self.activation_ctr = 0
        self.pooling_ctr = 0
        self.edram_wr_ctr = 0
        self.edram_rd_ctr = 0
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
            elif t == "edram_rd":
                self.edram_rd_ctr += 1
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
        print("edram_rd_ctr", self.edram_rd_ctr)
        print("data_transfer_ctr", self.data_transfer_ctr)
        print("total", len(self.Computation_order))

        if self.trace:
            layer = 0
            for e in self.Computation_order:
                if e.nlayer == layer:
                    layer += 1
                    print()
                    print("layer:", e.nlayer)
                print(self.Computation_order.index(e), e)

    def __str__(self):
        return str(self.__dict__)
