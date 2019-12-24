from HardwareMetaData import HardwareMetaData
from configs.ModelConfig import ModelConfig
from PE import PE

from EventMetaData import EventMetaData
from FetchEvent import FetchEvent
from Interconnect import Interconnect
from Packet import Packet

import numpy as np
from math import ceil, floor
import os, csv, copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from multiprocessing import Process
import time

class Controller(object):
    def __init__(self, ordergenerator, trace, mapping_str, scheduling_str):
        self.ordergenerator = ordergenerator
        self.trace = trace
        self.mapping_str = mapping_str
        self.scheduling_str = scheduling_str
        if self.scheduling_str == "Pipeline":
            self.isPipeLine = True
        else:
            self.isPipeLine = False
        self.Computation_order = self.ordergenerator.Computation_order
        print("Computation order length:", len(self.Computation_order))
        self.hd_info = HardwareMetaData()
        self.input_bit = self.ordergenerator.model_info.input_bit
        self.cycle_ctr = 0
        self.edram_read_cycles = \
            ceil(self.hd_info.Xbar_num * self.hd_info.Xbar_h * \
            self.input_bit * self.hd_info.eDRAM_read_latency / \
            self.hd_info.cycle_time)
        self.done_event = 0

        self.PE_array = []
        for rty_idx in range(self.hd_info.Router_num_y):
            for rtx_idx in range(self.hd_info.Router_num_x):
                for pey_idx in range(self.hd_info.PE_num_y):
                    for pex_idx in range(self.hd_info.PE_num_x):
                        pe_pos = (rty_idx, rtx_idx, pey_idx, pex_idx)
                        pe = PE(pe_pos, ModelConfig().input_bit)
                        self.PE_array.append(pe)

        # Energy
        self.Total_energy_edram_buffer = 0
        self.Total_energy_bus = 0
        self.Total_energy_router = 0
        self.Total_energy_activation = 0
        self.Total_energy_pe_shift_and_add = 0
        self.Total_energy_cu_shift_and_add = 0
        self.Total_energy_pooling = 0
        self.Total_energy_or = 0
        self.Total_energy_adc = 0
        self.Total_energy_dac = 0
        self.Total_energy_crossbar = 0
        self.Total_energy_ir_in_cu = 0
        self.Total_energy_or_in_cu = 0
        self.Total_energy_interconnect = 0

        # Interconnect
        self.fetch_array = []
        self.interconnect = Interconnect(self.hd_info.Router_num_y, self.hd_info.Router_num_x, self.input_bit)
        self.interconnect_step = self.hd_info.Router_flit_size / self.input_bit * self.hd_info.cycle_time * self.hd_info.Frequency # scaling from ISAAC
        self.interconnect_step = floor(self.interconnect_step)
        self.data_transfer_trigger = []
        self.data_transfer_erp = []
        # Pipeline control
        if not self.isPipeLine:
            self.pipeline_layer_stage = 0
            self.pipeline_stage_record = []
            self.events_each_layer = []
            for layer in range(self.ordergenerator.model_info.layer_length):
                self.events_each_layer.append(0)
            for e in self.Computation_order:
                self.events_each_layer[e.nlayer] += 1
            self.this_layer_event_ctr = 0
            self.this_layer_cycle_ctr = 0
            self.cycles_each_layer = []
            print("events_each_layer:", self.events_each_layer)

        # Statistics
        self.color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        self.mem_acc_ctr = 0
        self.data_transfer_ctr = 0
        self.act_xb_ctr = 0
        self.pe_saa_stall_cycle = 0

        # Utilization
        self.pe_state_for_plot = [[], []]
        self.cu_state_for_plot = [[], []]
        self.xb_state_for_plot = [[], []]
        self.buffer_size = []
        self.buffer_size_i = []
        # self.energy_utilization = []
        for i in range(len(self.PE_array)):
            self.buffer_size.append([])
            self.buffer_size_i.append([])
        self.max_buffer_size = 0 # num of data
        self.max_buffer_size_i = 0 # num of data

        # execute event
        self.erp_rd     = []
        self.erp_cu_op  = []
        self.erp_pe_saa = []
        self.erp_act    = []
        self.erp_wr     = []
        self.erp_pool   = []

        # Trigger pe
        self.trigger_edram_rd = []
        self.trigger_cu_op    = []
        self.trigger_pe_saa   = [] 
        self.trigger_edram_wr = []
        self.trigger_act      = []
        self.trigger_pool_rd  = []
        self.trigger_pool     = []

        self.buffer_record_freq =  100

    def run(self):
        # 只檢查有event的pe
        #pe_edram = []
        for e in self.Computation_order:
            if e.event_type == 'edram_rd_ir':
                if e.preceding_event_count == e.current_number_of_preceding_event:
                    pos = e.position_idx
                    rty, rtx, pey, pex, cuy, cux = pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]
                    pe_idx = pex + pey * self.hd_info.PE_num_x + rtx * self.hd_info.PE_num + rty * self.hd_info.Router_num_x * self.hd_info.PE_num
                    cu_idx = cux + cuy * self.hd_info.CU_num_x
                    pe = self.PE_array[pe_idx]
                    pe.CU_array[cu_idx].edram_rd_ir_erp.append(e)
                    if cu_idx not in pe.idle_eventQueuing_CU:
                        pe.idle_eventQueuing_CU.append(cu_idx)
                    if pe not in self.erp_rd:
                        self.erp_rd.append(pe)
                    # if pe not in pe_edram:
                    #     pe_edram.append(pe)
            if e.nlayer != 0:
                break

        t_edram = 0
        t_cuop = 0
        t_pesaa = 0
        t_act = 0
        t_wr = 0
        t_it = 0
        t_fe = 0
        t_tr = 0
        t_st = 0
        t_buf = 0
        t_poo = 0
        t_ = time.time()
        start_time = time.time()
        layer = 0
        while True:
            if self.cycle_ctr % 5000 == 0:
                if self.done_event == 0:
                    print("0")
                else:
                    print("layer:", layer)
                    print("Cycle",self.cycle_ctr, "Done event:", self.done_event, "time per event", (time.time()-start_time)/self.done_event, "time per cycle", (time.time()-start_time)/self.cycle_ctr)
                    print("edram:", t_edram, "t_cuop", t_cuop, "pesaa", t_pesaa, "act", t_act, "wr", t_wr)
                    print("iterconeect", t_it, "fetch", t_fe, "trigger", t_tr, "state", t_st)
                    print("buffer", t_buf, "pool", t_poo)
                    print("t:", time.time()-t_)
                    t_edram, t_cuop, t_pesaa, t_act, t_wr = 0, 0, 0, 0, 0
                    t_it, t_fe, t_tr, t_st, t_buf, t_poo = 0, 0, 0, 0, 0, 0
                    t_ = time.time()

            self.cycle_ctr += 1
            self.act_xb_ctr = 0
            if not self.isPipeLine:
                self.this_layer_cycle_ctr += 1
            if self.trace:
                print("cycle:", self.cycle_ctr)

            staa = time.time()
            self.event_edram_rd()
            t_edram += time.time() - staa

            staa = time.time()
            self.event_cu_op()
            t_cuop += time.time() - staa

            staa = time.time()
            self.event_pe_saa()
            t_pesaa += time.time() - staa

            staa = time.time()
            self.event_act()
            t_act += time.time() - staa

            staa = time.time()
            self.event_edram_wr()
            t_wr += time.time() - staa

            staa = time.time()
            self.event_edram_rd_pool()
            self.event_pool()
            t_poo += time.time() - staa
            
            staa = time.time()
            self.process_interconnect()
            t_it += time.time() - staa

            staa = time.time()
            self.event_transfer()
            t_tr += time.time() - staa

            staa = time.time()
            self.fetch()
            t_fe += time.time() - staa

            ### Pipeline stage control ###
            if not self.isPipeLine:
                self.pipeline_stage_record.append(self.pipeline_layer_stage)
                if self.this_layer_event_ctr == self.events_each_layer[self.pipeline_layer_stage]:
                    layer += 1
                    self.pipeline_layer_stage += 1
                    self.cycles_each_layer.append(self.this_layer_cycle_ctr)
                    self.this_layer_event_ctr = 0
                    self.this_layer_cycle_ctr = 0

            staa = time.time()
            self.trigger()
            t_tr += time.time() - staa

            ### Record PE State ###
            staa = time.time()
            for pe in self.PE_array:
                if pe.state:
                    self.pe_state_for_plot[0].append(self.cycle_ctr)
                    self.pe_state_for_plot[1].append(self.PE_array.index(pe))
            t_st += time.time() - staa

            ### Reset PE ###
            for pe in self.PE_array:
                pe.state = False

            ### Buffer utilization
            staa = time.time()
            if self.cycle_ctr % self.buffer_record_freq == 0:
                self.buffer_util()

            # procs = []
            # #for pe in self.PE_array:
            # for i in range(0, 4):
            #     proc = Process(target=self.job, args=(i,))
            #     procs.append(proc)
            #     proc.start()
            # # complete the processes
            # for proc in procs:
            #     proc.join()
            t_buf += time.time() - staa

            ### Finish
            if self.done_event == len(self.Computation_order):
                break

    def buffer_util(self):
        pe_idx = 0
        for pe in self.PE_array:
            bs   = len(pe.edram_buffer.buffer)
            bs_i = len(pe.edram_buffer_i.buffer)
            self.buffer_size  [pe_idx].append(bs)
            self.buffer_size_i[pe_idx].append(bs_i)
            self.max_buffer_size   = max(bs,   self.max_buffer_size)
            self.max_buffer_size_i = max(bs_i, self.max_buffer_size_i)
            pe_idx += 1

    def event_edram_rd(self):
        erp = []
        for pe in self.PE_array:
            if not pe.state_edram_rd_ir: # PE可以read, PE一次只做一個 Edram read to IR)
                if not pe.idle_eventQueuing_CU: # 沒有任何CU要做edram read
                    pass
                else: # 有Event正在排隊且為idle的CU
                    cu_idx = pe.idle_eventQueuing_CU.popleft()
                    cu = pe.CU_array[cu_idx]
                    event = cu.edram_rd_ir_erp.popleft()
                    pe.state_edram_rd_ir = True # 此PE正在執行edram read
                    pe.edram_rd_event = event # PE正在read此event

                    ## 開始做cu的edram read
                    ## 1. 資料是否還在transfer？
                    if event.data_is_transfer != 0:
                        continue
                    ## 2. 檢查資料是否都已在buffer中？
                    isData_ready = True
                    for inp in event.inputs: 
                        data = inp[1:] # inp: [num_input, fm_h, fm_w, fm_c]
                        if not pe.edram_buffer.check([event.nlayer, data]): # Data not in buffer
                            isData_ready = False
                            break
                    if not isData_ready: # 資料不在buffer, 則fetch from off-chip
                        event.data_is_transfer += len(event.inputs)
                        self.fetch_array.append(FetchEvent(event))
                        pe.data_is_fetching = True 
                        continue

            else: # 正在做某個cu的 edram read, 可能正在fetch from off-chip或是已經開始read了
                event = pe.edram_rd_event
                if event.data_is_transfer != 0:
                    continue
                pe.state = True
                if pe.edram_rd_cycle_ctr == 0:
                    if self.trace:
                        print("\tdo edram_rd_ir, nlayer:", event.nlayer,", pos:", event.position_idx, ",order index:", self.Computation_order.index(event))

                    # 計算耗能
                    num_data = len(event.inputs)
                    energy_edram_buffer = self.hd_info.Energy_edram_buffer * self.input_bit * num_data
                    energy_bus = self.hd_info.Energy_bus * self.input_bit * num_data
                    energy_ir_in_cu = self.hd_info.Energy_ir_in_cu * self.input_bit * num_data
                    self.Total_energy_edram_buffer += energy_edram_buffer
                    self.Total_energy_bus += energy_bus
                    self.Total_energy_ir_in_cu += energy_ir_in_cu
                    pe.Edram_buffer_energy += energy_edram_buffer
                    pe.Bus_energy += energy_bus
                    pe.CU_energy += energy_ir_in_cu

                pe.edram_rd_cycle_ctr += 1
                if pe.edram_rd_cycle_ctr == self.edram_read_cycles: # finish edram read
                    self.done_event += 1
                    if not self.isPipeLine:
                        self.this_layer_event_ctr += 1
                    pe.edram_rd_cycle_ctr = 0
                    pe.state_edram_rd_ir = False

                    # trigger cu operation
                    proceeding_index = event.proceeding_event[0] # 只會trigger一個cu operation
                    pro_event = self.Computation_order[proceeding_index]
                    pro_event.current_number_of_preceding_event += 1
                    pe.cu_op_trigger = pro_event
                    if pe not in self.trigger_cu_op:
                        self.trigger_cu_op.append(pe)

                    '''
                    # Free buffer (ideal)
                    pe_id = self.PE_array.index(pe)
                    nlayer = event.nlayer
                    if self.ordergenerator.model_info.layer_list[nlayer].layer_type == "convolution":
                        for d in event.inputs:
                            pos = d[2] + d[1]*self.ordergenerator.model_info.input_w[nlayer] + d[3]*self.ordergenerator.model_info.input_w[nlayer]*self.ordergenerator.model_info.input_h[nlayer] # w + h*width + c*height*width
                            self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] -= 1
                            if self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] == 0:
                                data = d[1:]
                                self.PE_array[pe_id].edram_buffer_i.buffer.remove([nlayer, data])
                    elif self.ordergenerator.model_info.layer_list[nlayer].layer_type == "fully":
                        for d in event.inputs:
                            pos = d[1]
                            self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] -= 1
                            if self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] == 0:
                                data = d[1:]
                                self.PE_array[pe_id].edram_buffer_i.buffer.remove([nlayer, data])
                    '''
            
            if pe.state_edram_rd_ir or pe.idle_eventQueuing_CU:
                erp.append(pe)
        self.erp_rd = erp

    def event_cu_op(self):
        erp = []
        for pe in self.erp_cu_op:
            for cu in pe.cu_op_list.copy():
                cu.state = False # for bottleneck analysis
                if cu.finish_cycle == 0 and cu.cu_op_event != 0: # cu operation start
                    if self.trace:
                        pass
                        print("\tcu operation start")
                    cu.finish_cycle = self.cycle_ctr - 1 + cu.cu_op_event.inputs + 2 # +2: pipeline 最後兩個 stage

                    ## Energy
                    ou_num_dict = cu.cu_op_event.outputs
                    for xb_idx in ou_num_dict:
                        ou_num = ou_num_dict[xb_idx]
                        self.Total_energy_crossbar += self.hd_info.Energy_ou_crossbar * ou_num
                        self.Total_energy_adc += self.hd_info.Energy_ou_dac * ou_num
                        self.Total_energy_cu_shift_and_add += self.hd_info.Energy_ou_ssa * ou_num

                    ## State
                    pe.state = True
                    cu.state = True
                    cu_id = self.PE_array.index(pe) * self.hd_info.CU_num + pe.CU_array.index(cu)
                    self.cu_state_for_plot[0].append(self.cycle_ctr)
                    self.cu_state_for_plot[1].append(cu_id)
                    for xb_idx in ou_num_dict:
                        xb_id = cu_id * self.hd_info.Xbar_num + xb_idx
                        ou_num = ou_num_dict[xb_idx]
                        for c in range(ou_num):
                            self.xb_state_for_plot[0].append(self.cycle_ctr+c)
                            self.xb_state_for_plot[1].append(xb_id)

                elif cu.finish_cycle != 0 and cu.cu_op_event != 0: ## doing cu operation
                    if cu.finish_cycle == self.cycle_ctr: # finish
                        if self.trace:
                            pass
                            print("\tcu operation finish")
                        self.done_event += 1
                        if not self.isPipeLine:
                            self.this_layer_event_ctr += 1
                        cu_idx = pe.CU_array.index(cu)
                        if cu.edram_rd_ir_erp:
                            if cu_idx not in pe.idle_eventQueuing_CU:
                                pe.idle_eventQueuing_CU.append(cu_idx)

                        ### add next event counter: pe_saa
                        for proceeding_index in cu.cu_op_event.proceeding_event:
                            pro_event = self.Computation_order[proceeding_index]
                            pro_event.current_number_of_preceding_event += 1
                            if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                if self.trace:
                                    pass
                                    #print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx)
                                if pro_event.event_type == "pe_saa":
                                    pe.pe_saa_trigger.append([pro_event, []])
                                    if pe not in self.trigger_pe_saa:
                                        self.trigger_pe_saa.append(pe)
                                elif pro_event.event_type == "edram_wr":
                                    pe.edram_wr_trigger.append([pro_event, []])
                                    if pe not in self.trigger_edram_wr:
                                        self.trigger_edram_wr.append(pe)

                        cu.finish_cycle = 0
                        cu.cu_op_event = 0
                        pe.cu_op_list.remove(cu)

                        ## State
                        pe.state = True
                        cu.state = True
                        cu_id = self.PE_array.index(pe) * self.hd_info.CU_num + pe.CU_array.index(cu)
                        self.cu_state_for_plot[0].append(self.cycle_ctr)
                        self.cu_state_for_plot[1].append(cu_id)
                    else:
                        if self.trace:
                            pass
                            print("\tcu operation")
                        ## State
                        pe.state = True
                        cu.state = True
                        cu_id = self.PE_array.index(pe) * self.hd_info.CU_num + pe.CU_array.index(cu)
                        self.cu_state_for_plot[0].append(self.cycle_ctr)
                        self.cu_state_for_plot[1].append(cu_id)

                # bottleneck analysis
                if not cu.state: # CU idle
                    if not cu.edram_rd_ir_erp: # CU idle, 且event pool是空的
                        cu.pure_idle_time += 1
                    else: # CU idle, 有event在event pool裡面, 這些event不能開始做, 肯定是資料沒到的拉
                        cu.wait_transfer_time += 1
                else: # CU busy
                    isEventWaiting = False
                    for event in cu.edram_rd_ir_erp:
                        if event.data_is_transfer == 0: # 有event資料已經ready但還不能做
                            isEventWaiting = True
                            break
                    if isEventWaiting:
                        cu.wait_resource_time += 1
                    else:
                        cu.pure_computation_time += 1
            if pe.cu_op_list:
                erp.append(pe)
        self.erp_cu_op = erp

    def event_pe_saa(self):
        erp = []
        for pe in self.erp_pe_saa:
            pe_saa_erp = []
            for event in pe.pe_saa_erp: # 1個cycle全部做完
                if event.data_is_transfer != 0: # 此event的資料正在傳輸
                    pe_saa_erp.append(event)
                    continue
                if self.trace:
                    pass
                    print("\tdo pe_saa, pe_pos:", pe.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event)) 
                self.done_event += 1
                if not self.isPipeLine:
                    self.this_layer_event_ctr += 1
                
                pe.state = True
                #pe.pe_saa_erp.remove(event)
                saa_amount = len(event.inputs[0]) # inputs[0]: 前面的cu index
                rm_data_list = event.inputs[1]    # 要移除的資料

                self.Total_energy_or += self.hd_info.Energy_or * self.input_bit * saa_amount
                self.Total_energy_bus += self.hd_info.Energy_bus * self.input_bit * saa_amount
                self.Total_energy_pe_shift_and_add += self.hd_info.Energy_shift_and_add * saa_amount
                self.Total_energy_bus += self.hd_info.Energy_bus * self.input_bit * saa_amount
                self.Total_energy_or += self.hd_info.Energy_or * self.input_bit * saa_amount

                pe.Or_energy += self.hd_info.Energy_or * self.input_bit * saa_amount
                pe.Bus_energy += self.hd_info.Energy_bus * self.input_bit * saa_amount
                pe.Shift_and_add_energy += self.hd_info.Energy_shift_and_add * saa_amount
                pe.Bus_energy += self.hd_info.Energy_bus * self.input_bit * saa_amount
                pe.Or_energy += self.hd_info.Energy_or * self.input_bit * saa_amount

                ### add next event counter: activation
                for proceeding_index in event.proceeding_event:
                    pro_event = self.Computation_order[proceeding_index]
                    pro_event.current_number_of_preceding_event += 1
                    
                    if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                        if self.trace:
                            pass
                            #print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx)
                        pe.activation_trigger.append([pro_event, []])
                        if pe not in self.trigger_act:
                            self.trigger_act.append(pe)
                '''
                # Free buffer (ideal)
                for d in rm_data_list:
                    if [event.nlayer, d] in pe.edram_buffer_i.buffer:
                        pe.edram_buffer_i.buffer.remove([event.nlayer, d])
                '''

            pe.pe_saa_erp = pe_saa_erp
            if pe.pe_saa_erp:
                erp.append(pe)
            '''
            # bottleneck analysis
            if not pe.state_pe_saa[0]:
                # 因為都是從0開使使用, 0 idle代表全部idle
                if not pe.pe_saa_erp:
                    # pe saa idle, 且event pool是空的
                    pe.saa_pure_idle_time += 1
                else:
                    # pe saa idle, 有event在event pool裡面, 這些event不能開始做, 肯定是資料沒到的拉
                    pe.saa_wait_transfer_time += 1
            else:
                # SAA busy
                isEventWaiting = False
                for event in pe.pe_saa_erp:
                    if event.data_is_transfer == 0:
                        # 有event資料已經ready但還不能做
                        isEventWaiting = True
                        break
                if isEventWaiting:
                    pe.saa_wait_resource_time += 1
                else:
                    pe.saa_pure_computation_time += 1
            '''
        self.erp_pe_saa = erp

    def event_act(self):
        erp = []
        for pe in self.erp_act:
            if not pe.activation_erp:
                continue
            pe.state = True
            if pe.activation_epc <= len(pe.activation_erp):
                do_act_num = pe.activation_epc
            else:
                do_act_num = len(pe.activation_erp)
            #act_erp_copy = pe.activation_erp.copy()
            for idx in range(do_act_num):
                #event = act_erp_copy[idx]
                event = pe.activation_erp.popleft()
                if self.trace:
                    pass
                    print("\tdo activation, pe_pos:", pe.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                self.done_event += 1
                if not self.isPipeLine:
                    self.this_layer_event_ctr += 1

                energy_or = self.hd_info.Energy_or * self.input_bit
                energy_bus = self.hd_info.Energy_bus * self.input_bit
                energy_activation = self.hd_info.Energy_activation
                self.Total_energy_or += energy_or
                self.Total_energy_bus +=energy_bus
                self.Total_energy_activation += energy_activation
                pe.Or_energy += energy_or
                pe.Bus_energy += energy_bus
                pe.Activation_energy += energy_activation

                ### add next event counter: edram_wr
                for proceeding_index in event.proceeding_event:
                    pro_event = self.Computation_order[proceeding_index]
                    pro_event.current_number_of_preceding_event += 1
                    
                    if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                        if self.trace:
                            pass
                            #print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx, self.Computation_order.index(pro_event))
                        pe.edram_wr_trigger.append([pro_event, []])
                        if pe not in self.trigger_edram_wr:
                            self.trigger_edram_wr.append(pe)

            if pe.activation_erp:
                erp.append(pe)
        self.erp_act = erp

    def event_edram_wr(self):
        erp = []
        for pe in self.erp_wr:
            if not pe.edram_wr_erp:
                continue
            pe.state = True
            if pe.edram_wr_epc <= len(pe.edram_wr_erp):
                do_wr_num = pe.edram_wr_epc
            else:
                do_wr_num = len(pe.edram_wr_erp)
            for idx in range(do_wr_num):
                event = pe.edram_wr_erp.popleft()
                self.done_event += 1
                if self.trace:
                    pass
                    print("\tdo edram_wr, pe_pos:", pe.position, "layer:", event.nlayer, \
                    ",order index:", self.Computation_order.index(event), "data:", event.outputs)
                if not self.isPipeLine:
                    self.this_layer_event_ctr += 1
                
                energy_bus = self.hd_info.Energy_bus * self.input_bit
                energy_edram_buffer = self.hd_info.Energy_edram_buffer * self.input_bit

                self.Total_energy_bus += energy_bus
                self.Total_energy_edram_buffer += energy_edram_buffer

                pe.Bus_energy += energy_bus
                pe.Edram_buffer_energy += energy_edram_buffer

                if len(event.outputs[0]) == 4: #and event.outputs[0][3] == "u": # same layer transfer
                    pe.edram_buffer.put([event.nlayer, event.outputs[0]])
                    pe.edram_buffer_i.put([event.nlayer, event.outputs[0]])
                else:
                    pe.edram_buffer.put([event.nlayer+1, event.outputs[0]])
                    pe.edram_buffer_i.put([event.nlayer+1, event.outputs[0]])

                ### add next event counter: edram_rd_ir, edram_rd_pool, data_transfer
                for proceeding_index in event.proceeding_event:
                    pro_event = self.Computation_order[proceeding_index]
                    pro_event.current_number_of_preceding_event += 1
                    
                    if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                        if self.trace:
                            pass
                            #print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx)
                        pos = pro_event.position_idx
                        if pro_event.event_type == "edram_rd_ir":
                            cu_y, cu_x = pos[4], pos[5]
                            cu_idx = cu_x + cu_y * self.hd_info.CU_num_x
                            pe.edram_rd_ir_trigger.append([pro_event, [cu_idx]])
                            if pe not in self.trigger_edram_rd:
                                self.trigger_edram_rd.append(pe)
                        elif pro_event.event_type == "edram_rd_pool":
                            pe.edram_rd_pool_trigger.append([pro_event, []])
                            if pe not in self.trigger_pool_rd:
                                self.trigger_pool_rd.append(pe)
                        elif pro_event.event_type == "data_transfer":
                            self.data_transfer_trigger.append([pro_event, []])

            if pe.edram_wr_erp:
                erp.append(pe)
        self.erp_wr = erp

    def event_edram_rd_pool(self):
        for pe in self.PE_array:
            idx = 0
            for event in pe.edram_rd_pool_erp.copy():
                if event.data_is_transfer != 0: # 此event的資料正在傳輸
                    continue
                isData_ready = True
                for inp in event.inputs:
                    data = inp[1:]
                    #print(event.nlayer, data)
                    if not pe.edram_buffer.check([event.nlayer, data]):
                        # Data not in buffer
                        if self.trace:
                            print("\tData not ready for edram_rd_pool. Data: layer", event.nlayer, event.event_type, data)
                        isData_ready = False
                        pe.edram_rd_pool_erp.remove(event)
                        break

                if not isData_ready:
                    event.data_is_transfer += len(event.inputs)
                    self.fetch_array.append(FetchEvent(event))
                    continue
                self.done_event += 1
                if self.trace:
                    print("\tdo edram_rd_pool, pe_pos:", pe.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                if not self.isPipeLine:
                    self.this_layer_event_ctr += 1

                num_data = len(event.inputs)
                energy_edram_buffer = self.hd_info.Energy_edram_buffer * self.input_bit * num_data
                energy_bus = self.hd_info.Energy_bus * self.input_bit * num_data

                self.Total_energy_edram_buffer += energy_edram_buffer
                self.Total_energy_bus += self.hd_info.Energy_bus * self.input_bit * num_data

                pe.Edram_buffer_energy += energy_edram_buffer
                pe.Bus_energy += energy_bus

                #pe.state_edram_rd_pool[idx] = True
                pe.state = True
                pe.edram_rd_pool_erp.remove(event)
                idx += 1

                ### add next event counter: pooling
                for proceeding_index in event.proceeding_event:
                    pro_event = self.Computation_order[proceeding_index]
                    pro_event.current_number_of_preceding_event += 1

                    if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                        if self.trace:
                            pass
                            #print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx)
                        pos = pro_event.position_idx
                        pe.pooling_trigger.append([pro_event, []])
                        if pe not in self.trigger_pool:
                            self.trigger_pool.append(pe)
                '''
                # Free buffer (ideal)
                pe_id = self.PE_array.index(pe)
                nlayer = event.nlayer
                for d in event.inputs:
                    pos = d[2] + d[1]*self.ordergenerator.model_info.input_w[nlayer] + d[3]*self.ordergenerator.model_info.input_w[nlayer]*self.ordergenerator.model_info.input_h[nlayer] # w + h*width + c*height*width
                    self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] -= 1
                    if self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] == 0:
                        data = d[1:]
                        self.PE_array[pe_id].edram_buffer_i.buffer.remove([nlayer, data])
                '''
                if idx >= pe.edram_rd_pool_epc:
                    break # 一次讀len(pe.state_edram_rd_pool)筆
            
            '''
            # bottleneck analysis
            if not pe.state_edram_rd_pool:
                # Pooling unit idle
                if not pe.edram_rd_pool_erp:
                    # Pooling unit idle, 且event pool是空的
                    pe.pooling_pure_idle_time += 1
                else:
                    # Pooling unit idle, 有event在event pool裡面, 這些event不能開始做, 因為資料沒到
                    pe.pooling_wait_transfer_time += 1
            else:
                # Pooling unit busy
                isEventWaiting = False
                for event in pe.edram_rd_pool_erp:
                    if event.data_is_transfer == 0:
                        # 有event資料已經ready但還不能做
                        isEventWaiting = True
                        break
                if isEventWaiting:
                    pe.pooling_wait_resource_time += 1
                else:
                    pe.pooling_pure_computation_time += 1
            '''

    def event_pool(self):
        erp = []
        for pe in self.erp_pool:
            if not pe.pooling_erp:
                continue
            pe.state = True
            if pe.pooling_epc <= len(pe.pooling_erp):
                do_pool_num = pe.pooling_epc
            else:
                do_pool_num = len(pe.pooling_erp)
            for idx in range(do_pool_num):
                event = pe.pooling_erp.popleft()
                self.done_event += 1
                if self.trace:
                    print("\tdo pooling, pe_pos:", pe.position, "layer:", event.nlayer, ",order index:", self.Computation_order.index(event))
                if not self.isPipeLine:
                    self.this_layer_event_ctr += 1

                self.Total_energy_pooling += self.hd_info.Energy_pooling

                pe.Pooling_energy += self.hd_info.Energy_pooling

                ### add next event counter: edram_wr
                for proceeding_index in event.proceeding_event:
                    pro_event = self.Computation_order[proceeding_index]
                    pro_event.current_number_of_preceding_event += 1

                    if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                        if self.trace:
                            pass
                            #print("\t\tProceeding event is triggered.", pro_event.event_type, pro_event.position_idx)
                        pe.edram_wr_trigger.append([pro_event, []])

            if pe.pooling_erp:
                erp.append(pe)
        self.erp_pool = erp

    def process_interconnect(self):
        for s in range(self.interconnect_step):
            self.interconnect.step()
            self.Total_energy_interconnect += self.interconnect.step_energy_consumption
        # Packets arrive: Store data, trigger event
        arrived_packet = self.interconnect.get_arrived_packet()
        for pk in arrived_packet:
            if self.trace:
                pass
                #print("\tArrived packet:", pk)
            rty, rtx = pk.destination[0], pk.destination[1]
            pey, pex = pk.destination[2], pk.destination[3]
            pe_idx = pex + pey * self.hd_info.PE_num_x + rtx * self.hd_info.PE_num + rty * self.hd_info.PE_num * self.hd_info.Router_num_x
            pe = self.PE_array[pe_idx]

            self.Total_energy_edram_buffer += self.hd_info.Energy_edram_buffer * self.input_bit # write
            pe.Edram_buffer_energy += self.hd_info.Energy_edram_buffer * self.input_bit

            pe.edram_buffer.put(pk.data)
            pe.edram_buffer_i.put(pk.data)
            if self.trace:
                pass
                #print("put packet data:", pk)

            for pro_event_idx in pk.pro_event_list:
                if not self.isPipeLine:
                    self.this_layer_event_ctr += 1

                pro_event = self.Computation_order[pro_event_idx]
                pro_event.data_is_transfer -= 1

    def event_transfer(self):
        for event in self.data_transfer_erp.copy():
            self.done_event += 1
            if self.trace:
                pass
                #print("\tdo data_transfer, layer:", event.nlayer, ",order index:", self.Computation_order.index(event), \
                #        "pos:", event.position_idx, "data:", event.outputs)
            self.data_transfer_erp.remove(event)

            src = event.position_idx[0]
            des = event.position_idx[1]

            pro_event_list = [event.proceeding_event[0]]
            if self.Computation_order[pro_event_list[0]].event_type == "edram_rd_ir":
                data = [event.nlayer+1, event.outputs[0]]
            elif self.Computation_order[pro_event_list[0]].event_type == "edram_rd_pool":
                data = [event.nlayer+1, event.outputs[0]]
            elif self.Computation_order[pro_event_list[0]].event_type == "pe_saa":
                data = [event.nlayer, event.outputs[0]]
            else:
                print("transfer proceeding event type error.\n exit.")
                exit()
            packet = Packet(src, des, data, pro_event_list)
            src_pe_id = src[3] + src[2] * self.hd_info.PE_num_x + \
                        src[1] * self.hd_info.PE_num + \
                        src[0] * self.hd_info.PE_num * self.hd_info.Router_num_x
            src_pe = self.PE_array[src_pe_id]

            self.interconnect.input_packet(packet)

            energy_edram_buffer = self.hd_info.Energy_edram_buffer * self.input_bit

            self.Total_energy_edram_buffer += energy_edram_buffer # read

            src_pe.Edram_buffer_energy += energy_edram_buffer

            ### add next event counter
            des_pe_id = des[3] + des[2] * self.hd_info.PE_num_x + \
                        des[1] * self.hd_info.PE_num + \
                        des[0] * self.hd_info.PE_num * self.hd_info.Router_num_x
            des_pe = self.PE_array[des_pe_id]
            for proceeding_index in event.proceeding_event: # 這個loop只會執行一次(一個transfer後都只接一個event)
                pro_event = self.Computation_order[proceeding_index]
                pro_event.current_number_of_preceding_event += 1
                pro_event.data_is_transfer += 1
                if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                    if pro_event.event_type == "edram_rd_ir":
                        cuy, cux = pro_event.position_idx[4], pro_event.position_idx[5]
                        cu_idx = cux + cuy * self.hd_info.CU_num_x
                        des_pe.edram_rd_ir_trigger.append([pro_event, [cu_idx]])
                        if des_pe not in self.trigger_edram_rd:
                            self.trigger_edram_rd.append(des_pe)
                    elif pro_event.event_type == "edram_rd_pool":
                        des_pe.edram_rd_pool_trigger.append([pro_event, []])
                        if des_pe not in self.trigger_pool_rd:
                            self.trigger_pool_rd.append(des_pe)
                    elif pro_event.event_type == "pe_saa":
                        des_pe.pe_saa_trigger.append([pro_event, []])
                        if des_pe not in self.trigger_pe_saa:
                            self.trigger_pe_saa.append(des_pe)
            '''
            # Free buffer (ideal)
            pe_id = src[3] + src[2]*self.hd_info.PE_num_x + \
                    src[1]*self.hd_info.PE_num + src[0]*self.hd_info.PE_num*self.hd_info.Router_num_x
            if len(event.outputs[0]) == 4:  #and event.outputs[0][3] == "u": # same layer data transfer
                data = event.outputs[0]
                if [event.nlayer, data] in self.PE_array[pe_id].edram_buffer_i.buffer:
                    self.PE_array[pe_id].edram_buffer_i.buffer.remove([event.nlayer, data])
            else:
                nlayer = event.nlayer+1
                if self.ordergenerator.model_info.layer_list[nlayer].layer_type != "fully":
                    for d in event.outputs:
                        pos = d[1] + d[0]*self.ordergenerator.model_info.input_w[nlayer] + d[2]*self.ordergenerator.model_info.input_w[nlayer]*self.ordergenerator.model_info.input_h[nlayer] # w + h*width + c*height*width
                        self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] -= 1
                        if self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] == 0:
                            self.PE_array[pe_id].edram_buffer_i.buffer.remove([nlayer, d])
                else:
                    for d in event.outputs:
                        pos = d[0]
                        self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] -= 1
                        if self.ordergenerator.free_buffer_controller.input_require[pe_id][nlayer][pos] == 0:
                            self.PE_array[pe_id].edram_buffer_i.buffer.remove([nlayer, d])
            '''

    def fetch(self):
        des_dict = dict()
        for FE in self.fetch_array.copy():
            FE.cycles_counter += 1
            if FE.cycles_counter == FE.fetch_cycle:
                src  = (0, FE.event.position_idx[1], -1, -1)
                des  = FE.event.position_idx[0:4]
                if FE.event.event_type == "edram_rd_ir":
                    if not self.isPipeLine:
                        self.this_layer_event_ctr -= len(FE.event.inputs)
                    FE.event.preceding_event_count += len(FE.event.inputs)
                    if des not in des_dict:
                        des_dict[des] = []
                    for inp in FE.event.inputs:
                        data = inp[1:]
                        data = [FE.event.nlayer, data]
                        pro_event_idx = self.Computation_order.index(FE.event)

                        isDataInDict = False
                        for data_pro_event in des_dict[des]:
                            if data == data_pro_event[0]:
                                data_pro_event[1].append(pro_event_idx)
                                isDataInDict = True
                                break
                        if not isDataInDict:
                            des_dict[des].append([data, [pro_event_idx]])
                elif FE.event.event_type == "edram_rd_pool":
                    if not self.isPipeLine:
                        self.this_layer_event_ctr -= len(FE.event.inputs)
                    FE.event.preceding_event_count += len(FE.event.inputs)
                    if des not in des_dict:
                        des_dict[des] = []
                    for data in FE.event.inputs:
                        data = [FE.event.nlayer, data]
                        pro_event_idx = self.Computation_order.index(FE.event)
                        isDataInDict = False
                        for data_pro_event in des_dict[des]:
                            if data == data_pro_event[0]:
                                data_pro_event[1].append(pro_event_idx)
                                isDataInDict = True
                                break
                        if not isDataInDict:
                            des_dict[des].append([data, [pro_event_idx]])
                self.fetch_array.remove(FE)
        for des in des_dict:
            src = (0, des[1], -1, -1)
            for data_pro_event in des_dict[des]:
                data, pro_event_idx = data_pro_event[0], data_pro_event[1]
                packet = Packet(src, des, data, pro_event_idx)
                self.interconnect.input_packet(packet)

    def trigger(self):
        ## Trigger interconnect
        for trigger in self.data_transfer_trigger:
            pro_event = trigger[0]
            self.data_transfer_erp.append(pro_event)
        self.data_transfer_trigger = []

        # Trigger edram_rd_ir
        if not self.isPipeLine:
            tri_edram = []
            for pe in self.trigger_edram_rd:
                for trigger in pe.edram_rd_ir_trigger.copy():
                    pro_event = trigger[0]
                    cu_idx    = trigger[1][0]
                    if pro_event.nlayer == self.pipeline_layer_stage:
                        pe.CU_array[cu_idx].edram_rd_ir_erp.append(pro_event)
                        if pe.CU_array[cu_idx].edram_rd_ir_erp:
                            if cu_idx not in pe.idle_eventQueuing_CU:
                                pe.idle_eventQueuing_CU.append(cu_idx)
                        pe.edram_rd_ir_trigger.remove(trigger)
                        append_event = True
                    else:
                        append_event = False
                if append_event:
                    if pe not in self.erp_rd:
                        self.erp_rd.append(pe)
                if pe.edram_rd_ir_trigger:
                    tri_edram.append(pe)
            self.trigger_edram_rd = tri_edram
        else: # pipeline
            for pe in self.trigger_edram_rd:
                for trigger in pe.edram_rd_ir_trigger:
                    pro_event = trigger[0]
                    cu_idx    = trigger[1][0]
                    pe.CU_array[cu_idx].edram_rd_ir_erp.append(pro_event)
                pe.edram_rd_ir_trigger = []
                if pe not in self.erp_rd:
                    self.erp_rd.append(pe)
            self.trigger_edram_rd = []

        for pe in self.trigger_cu_op:
            #if pe.cu_op_trigger != 0: # 應該不用判斷
            event = pe.cu_op_trigger
            cuy, cux = event.position_idx[4], event.position_idx[5]
            cu_idx = cux + cuy * self.hd_info.CU_num_x
            cu = pe.CU_array[cu_idx]
            cu.cu_op_event = event
            pe.cu_op_list.append(cu)
            pe.cu_op_trigger = 0
            if pe not in self.erp_cu_op:
                self.erp_cu_op.append(pe)
        self.trigger_cu_op = []

        for pe in self.trigger_pe_saa:
            for trigger in pe.pe_saa_trigger:
                pro_event = trigger[0]
                pe.pe_saa_erp.append(pro_event) 
            pe.pe_saa_trigger = []
            if pe not in self.erp_pe_saa:
                self.erp_pe_saa.append(pe)
        self.trigger_pe_saa = []

        for pe in self.trigger_act:
            for trigger in pe.activation_trigger:
                pro_event = trigger[0]
                pe.activation_erp.append(pro_event)
            pe.activation_trigger = []
            if pe not in self.erp_act:
                self.erp_act.append(pe)
        self.trigger_act = []

        for pe in self.trigger_edram_wr:
            for trigger in pe.edram_wr_trigger:
                pro_event = trigger[0]
                pe.edram_wr_erp.append(pro_event)
            pe.edram_wr_trigger = []
            if pe not in self.erp_wr:
                self.erp_wr.append(pe)
        self.trigger_edram_wr = []

        # Trigger edram_read_pool
        if not self.isPipeLine:
            tri_pool_rd = []
            for pe in self.trigger_pool_rd:
                for trigger in pe.edram_rd_pool_trigger.copy():
                    pro_event = trigger[0]
                    if pro_event.nlayer == self.pipeline_layer_stage:
                        pe.edram_rd_pool_erp.append(pro_event)
                        pe.edram_rd_pool_trigger.remove(trigger)
                if pe.edram_rd_pool_trigger:
                    tri_pool_rd.append(pe)
            self.trigger_pool_rd = tri_pool_rd
        else:
            for pe in self.trigger_pool_rd:
                for trigger in pe.edram_rd_pool_trigger:
                    pro_event = trigger[0]
                    pe.edram_rd_pool_erp.append(pro_event)
                pe.edram_rd_pool_trigger = []
            self.trigger_pool_rd = []
        
        ## Trigger pooling 
        for pe in self.trigger_pool:
            for trigger in pe.pooling_trigger:
                pro_event = trigger[0]
                pe.pooling_erp.append(pro_event)
            pe.pooling_trigger = []
            if pe not in self.erp_pool:
                self.erp_pool.append(pe)
        self.trigger_pool = []

    def print_statistics_result(self):
        print("Total Cycles:", self.cycle_ctr)
        if not self.isPipeLine:
            print("Cycles each layer:", self.cycles_each_layer)
        print("Cycles time:", self.hd_info.cycle_time, "ns\n")
        print()
        
        if not self.isPipeLine:
            self.non_pipeline_stage()

        self.freq = 1
        print("output buffer utilization...")
        self.buffer_analysis()
        self.energy_breakdown()
        print("output bottleneck anaylsis...")
        self.bottleneck_statistics()
        print("output pe utilization...")
        self.pe_utilization()
        print("output cu utilization...")
        self.cu_utilization()
        print("output xb utilization...")
        self.crossbar_utilization()

    def bottleneck_statistics(self):
        ## CU
        total_pe_num = len(self.PE_array)
        idx = np.arange(total_pe_num)
        pure_idle_time, wait_transfer_time, wait_resource_time, pure_computation_time = [], [], [], []
        pure_idle_time_total, wait_transfer_time_total, wait_resource_time_total, pure_computation_time_total = 0, 0, 0, 0
        for j in range(self.hd_info.CU_num):
            pure_idle_time.append([])
            wait_transfer_time.append([])
            wait_resource_time.append([])
            pure_computation_time.append([])
            for i in range(len(self.PE_array)):
                pure_idle_time[j].append(self.PE_array[i].CU_array[j].pure_idle_time)
                wait_transfer_time[j].append(self.PE_array[i].CU_array[j].wait_transfer_time)
                wait_resource_time[j].append(self.PE_array[i].CU_array[j].wait_resource_time)
                pure_computation_time[j].append(self.PE_array[i].CU_array[j].pure_computation_time)
                pure_idle_time_total += pure_idle_time[j][-1]
                wait_transfer_time_total += wait_transfer_time[j][-1]
                wait_resource_time_total += wait_resource_time[j][-1]
                pure_computation_time_total +=pure_computation_time[j][-1]
        pure_idle_time = np.array(pure_idle_time)
        wait_transfer_time = np.array(wait_transfer_time)
        wait_resource_time = np.array(wait_resource_time)
        pure_computation_time = np.array(pure_computation_time)
        for i in range(self.hd_info.CU_num):
            plt.bar(idx+0.2*i, pure_idle_time[i], width = 0.2, color='b', edgecolor = 'white')
            plt.bar(idx+0.2*i, wait_transfer_time[i], bottom=pure_idle_time[i], color='r',  width=0.2, edgecolor = 'white')
            plt.bar(idx+0.2*i, wait_resource_time[i], bottom=pure_idle_time[i]+wait_transfer_time[i], color='g', width=0.2, edgecolor = 'white')
            plt.bar(idx+0.2*i, pure_computation_time[i], bottom=pure_idle_time[i]+wait_transfer_time[i]+wait_resource_time[i], color='y', width=0.2, edgecolor = 'white')
        plt.legend(["pure_idle", "wait_transfer", "wait_resource", "pure_computation"])
        plt.title(self.mapping_str+", "+self.scheduling_str)
        plt.xlabel('PE index')
        plt.ylabel('Cycle')
        plt.savefig('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/Bottleneck_CU.png')
        plt.clf()

        print("CU Bottleneck analysis:")
        print("\tTotal pure_idle:", pure_idle_time_total)
        print("\tTotal wait_transfer:", wait_transfer_time_total)
        print("\tTotal wait_resource:", wait_resource_time_total)
        print("\tTotal pure_computation:", pure_computation_time_total)
        total_cu_num = total_pe_num * self.hd_info.CU_num
        print("\tAverage pure_idle, ", pure_idle_time_total/total_cu_num, end="")
        print("(" + str(pure_idle_time_total/total_cu_num/self.cycle_ctr*100) + "%)")
        print("\tAverage wait_transfer, ", wait_transfer_time_total/total_cu_num, end="")
        print("(" + str(wait_transfer_time_total/total_cu_num/self.cycle_ctr*100) + "%)")
        print("\tAverage wait_resource, ", wait_resource_time_total/total_cu_num, end="")
        print("(" + str(wait_resource_time_total/total_cu_num/self.cycle_ctr*100) + "%)")
        print("\tAverage pure_computation, ", pure_computation_time_total/total_cu_num, end="")
        print("(" + str(pure_computation_time_total/total_cu_num/self.cycle_ctr*100) + "%)")
        print()

        '''
        # PE SAA
        total_pe_num = len(self.PE_array)
        idx = np.arange(total_pe_num)
        pure_idle_time = list()
        wait_transfer_time = list()
        wait_resource_time = list()
        pure_computation_time = list()
        pure_idle_time_total = 0
        wait_transfer_time_total = 0
        wait_resource_time_total = 0
        pure_computation_time_total = 0
        for i in range(len(self.PE_array)):
            pure_idle_time.append(self.PE_array[i].saa_pure_idle_time)
            wait_transfer_time.append(self.PE_array[i].saa_wait_transfer_time)
            wait_resource_time.append(self.PE_array[i].saa_wait_resource_time)
            pure_computation_time.append(self.PE_array[i].saa_pure_computation_time)
            pure_idle_time_total += pure_idle_time[-1]
            wait_transfer_time_total += wait_transfer_time[-1]
            wait_resource_time_total += wait_resource_time[-1]
            pure_computation_time_total +=pure_computation_time[-1]
        pure_idle_time = np.array(pure_idle_time)
        wait_transfer_time = np.array(wait_transfer_time)
        wait_resource_time = np.array(wait_resource_time)
        pure_computation_time = np.array(pure_computation_time)
        plt.bar(idx, pure_idle_time, color='b',  width=0.8, edgecolor = 'white')
        plt.bar(idx, wait_transfer_time, bottom=pure_idle_time, color='r',  width=0.8, edgecolor = 'white')
        plt.bar(idx, wait_resource_time, bottom=pure_idle_time+wait_transfer_time, color='g',  width=0.8, edgecolor = 'white')
        plt.bar(idx, pure_computation_time, bottom=pure_idle_time+wait_transfer_time+wait_resource_time, color='y',  width=0.8, edgecolor = 'white')
        plt.legend(["pure_idle", "wait_transfer", "wait_resource", "pure_computation"])
        plt.title(self.mapping_str+", "+self.scheduling_str)
        plt.xlabel('PE index')
        plt.ylabel('Cycle')
        plt.savefig('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/Bottleneck_PE_SAA.png')
        plt.clf()
        print("PE SAA Bottleneck analysis:")
        print("\tTotal pure_idle:", pure_idle_time_total)
        print("\tTotal wait_transfer:", wait_transfer_time_total)
        print("\tTotal wait_resource:", wait_resource_time_total)
        print("\tTotal pure_computation:", pure_computation_time_total)
        print("\tAverage pure_idle, ", pure_idle_time_total/total_pe_num, end="")
        print("(" + str(pure_idle_time_total/total_pe_num/self.cycle_ctr*100) + "%)")
        print("\tAverage wait_transfer, ", wait_transfer_time_total/total_pe_num, end="")
        print("(" + str(wait_transfer_time_total/total_pe_num/self.cycle_ctr*100) + "%)")
        print("\tAverage wait_resource, ", wait_resource_time_total/total_pe_num, end="")
        print("(" + str(wait_resource_time_total/total_pe_num/self.cycle_ctr*100) + "%)")
        print("\tAverage pure_computation, ", pure_computation_time_total/total_pe_num, end="")
        print("(" + str(pure_computation_time_total/total_pe_num/self.cycle_ctr*100) + "%)")
        print()

        # Pooling
        total_pe_num = len(self.PE_array)
        idx = np.arange(total_pe_num)
        pure_idle_time = list()
        wait_transfer_time = list()
        wait_resource_time = list()
        pure_computation_time = list()
        pure_idle_time_total = 0
        wait_transfer_time_total = 0
        wait_resource_time_total = 0
        pure_computation_time_total = 0
        for i in range(len(self.PE_array)):
            pure_idle_time.append(self.PE_array[i].pooling_pure_idle_time)
            wait_transfer_time.append(self.PE_array[i].pooling_wait_transfer_time)
            wait_resource_time.append(self.PE_array[i].pooling_wait_resource_time)
            pure_computation_time.append(self.PE_array[i].pooling_pure_computation_time)
            pure_idle_time_total += pure_idle_time[-1]
            wait_transfer_time_total += wait_transfer_time[-1]
            wait_resource_time_total += wait_resource_time[-1]
            pure_computation_time_total +=pure_computation_time[-1]
        pure_idle_time = np.array(pure_idle_time)
        wait_transfer_time = np.array(wait_transfer_time)
        wait_resource_time = np.array(wait_resource_time)
        pure_computation_time = np.array(pure_computation_time)
        plt.bar(idx, pure_idle_time, color='b',  width=0.8, edgecolor = 'white')
        plt.bar(idx, wait_transfer_time, bottom=pure_idle_time, color='r',  width=0.8, edgecolor = 'white')
        plt.bar(idx, wait_resource_time, bottom=pure_idle_time+wait_transfer_time, color='g',  width=0.8, edgecolor = 'white')
        plt.bar(idx, pure_computation_time, bottom=pure_idle_time+wait_transfer_time+wait_resource_time, color='y',  width=0.8, edgecolor = 'white')
        plt.legend(["pure_idle", "wait_transfer", "wait_resource", "pure_computation"])
        plt.title(self.mapping_str+", "+self.scheduling_str)
        plt.xlabel('PE index')
        plt.ylabel('Cycle')
        plt.savefig('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/Bottleneck_Pooling.png')
        plt.clf()
        print("Pooling Bottleneck analysis:")
        print("\tTotal pure_idle:", pure_idle_time_total)
        print("\tTotal wait_transfer:", wait_transfer_time_total)
        print("\tTotal wait_resource:", wait_resource_time_total)
        print("\tTotal pure_computation:", pure_computation_time_total)
        print("\tAverage pure_idle, ", pure_idle_time_total/total_pe_num, end="")
        print("(" + str(pure_idle_time_total/total_pe_num/self.cycle_ctr*100) + "%)")
        print("\tAverage wait_transfer, ", wait_transfer_time_total/total_pe_num, end="")
        print("(" + str(wait_transfer_time_total/total_pe_num/self.cycle_ctr*100) + "%)")
        print("\tAverage wait_resource, ", wait_resource_time_total/total_pe_num, end="")
        print("(" + str(wait_resource_time_total/total_pe_num/self.cycle_ctr*100) + "%)")
        print("\tAverage pure_computation, ", pure_computation_time_total/total_pe_num, end="")
        print("(" + str(pure_computation_time_total/total_pe_num/self.cycle_ctr*100) + "%)")
        print()
        '''

    def energy_breakdown(self):
        self.Total_energy_cu = self.Total_energy_cu_shift_and_add + \
                                self.Total_energy_adc + \
                                self.Total_energy_dac + \
                                self.Total_energy_crossbar + \
                                self.Total_energy_ir_in_cu + \
                                self.Total_energy_or_in_cu
        self.Total_energy_pe = self.Total_energy_cu + \
                                self.Total_energy_edram_buffer + \
                                self.Total_energy_bus + \
                                self.Total_energy_activation + \
                                self.Total_energy_pe_shift_and_add + \
                                self.Total_energy_pooling + \
                                self.Total_energy_or
        self.Total_energy = self.Total_energy_pe + self.Total_energy_interconnect

        print("Energy breakdown:")
        print("\tTotal:", self.Total_energy, "nJ")
        print("\tChip level")
        print("\t\tPE: %.4e (%.2f%%)" %(self.Total_energy_pe, self.Total_energy_pe/self.Total_energy*100))
        print("\t\tInterconnect: %.4e (%.2f%%)" %(self.Total_energy_interconnect, self.Total_energy_interconnect/self.Total_energy*100))
        print()
        print("\tPE level")
        print("\t\tCU: %.4e (%.2f%%)" %(self.Total_energy_cu, self.Total_energy_cu/self.Total_energy_pe*100))
        print("\t\tEdram Buffer: %.4e (%.2f%%)" %(self.Total_energy_edram_buffer, self.Total_energy_edram_buffer/self.Total_energy_pe*100))
        print("\t\tBus: %.4e (%.2f%%)" %(self.Total_energy_bus, self.Total_energy_bus/self.Total_energy_pe*100))
        print("\t\tActivation: %.4e (%.2f%%)" %(self.Total_energy_activation, self.Total_energy_activation/self.Total_energy_pe*100))
        print("\t\tShift and Add: %.4e (%.2f%%)" %(self.Total_energy_pe_shift_and_add, self.Total_energy_pe_shift_and_add/self.Total_energy_pe*100))
        print("\t\tPooling: %.4e (%.2f%%)" %(self.Total_energy_pooling, self.Total_energy_pooling/self.Total_energy_pe*100))
        print("\t\tOR: %.4e (%.2f%%)" %(self.Total_energy_or, self.Total_energy_or/self.Total_energy_pe*100))
        print()
        print("\tCU level")
        print("\t\tShift and Add: %.4e (%.2f%%)" %(self.Total_energy_cu_shift_and_add, self.Total_energy_cu_shift_and_add/self.Total_energy_cu*100))
        print("\t\tADC: %.4e (%.2f%%)" %(self.Total_energy_adc, self.Total_energy_adc/self.Total_energy_cu*100))
        print("\t\tDAC: %.4e (%.2f%%)" %(self.Total_energy_dac, self.Total_energy_dac/self.Total_energy_cu*100))
        print("\t\tCrossbar Array: %.4e (%.2f%%)" %(self.Total_energy_crossbar, self.Total_energy_crossbar/self.Total_energy_cu*100))
        print("\t\tIR: %.4e (%.2f%%)" %(self.Total_energy_ir_in_cu, self.Total_energy_ir_in_cu/self.Total_energy_cu*100))
        print("\t\tOR: %.4e (%.2f%%)" %(self.Total_energy_or_in_cu, self.Total_energy_or_in_cu/self.Total_energy_cu*100))
        print()

        # Chip pie
        labels = 'PE', 'Interconnect'
        value = [self.Total_energy_pe, self.Total_energy_interconnect]
        plt.pie(value , labels = labels, autopct='%1.1f%%')
        plt.axis('equal')
        plt.title(self.mapping_str+", "+self.scheduling_str)
        plt.savefig('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/Energy_breakdown_chip.png')
        plt.clf()

        # PE breakdown
        total_pe_num = len(self.PE_array)
        idx = np.arange(total_pe_num)
        CU_energy, Edram_buffer_energy = list(), list()
        Bus_energy, Shift_and_add_energy = list(), list()
        Or_energy, Activation_energy, Pooling_energy = list(), list(), list()
        for pe in self.PE_array:
            CU_energy.append(pe.CU_energy)
            Edram_buffer_energy.append(pe.Edram_buffer_energy)
            Bus_energy.append(pe.Bus_energy)
            Shift_and_add_energy.append(pe.Shift_and_add_energy)
            Or_energy.append(pe.Or_energy)
            Activation_energy.append(pe.Activation_energy)
            Pooling_energy.append(pe.Pooling_energy)
        CU_energy, Edram_buffer_energy = np.array(CU_energy), np.array(Edram_buffer_energy)
        Bus_energy, Shift_and_add_energy = np.array(Bus_energy), np.array(Shift_and_add_energy)
        Or_energy, Activation_energy = np.array(Or_energy), np.array(Activation_energy)
        Pooling_energy = np.array(Pooling_energy)

        plt.bar(idx, CU_energy,  width=0.8)
        energy_sum = CU_energy
        plt.bar(idx, Edram_buffer_energy, bottom=energy_sum,  width=0.8)
        energy_sum += Edram_buffer_energy
        plt.bar(idx, Bus_energy, bottom=energy_sum,  width=0.8)
        energy_sum += Bus_energy
        plt.bar(idx, Shift_and_add_energy, bottom=energy_sum,  width=0.8)
        energy_sum += Shift_and_add_energy
        plt.bar(idx, Or_energy, bottom=energy_sum,  width=0.8)
        energy_sum += Or_energy
        plt.bar(idx, Activation_energy, bottom=energy_sum,  width=0.8)
        energy_sum += Activation_energy
        plt.bar(idx, Pooling_energy, bottom=energy_sum,  width=0.8)
        plt.xlabel('PE index')
        plt.ylabel('Energy (nJ)')
        plt.legend(["CU_energy", "Edram_buffer_energy", "Bus_energy", "Shift_and_add_energy", "Or_energy", "Activation_energy", "Pooling_energy"])
        plt.savefig('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/Energy_breakdown_PE.png')
        plt.clf()

        # PE breakdown (percentage)
        for i in range(len(self.PE_array)):
            sum_energy = CU_energy[i] + Edram_buffer_energy[i] + Bus_energy[i] + \
                         Shift_and_add_energy[i] + Or_energy[i] + \
                         Activation_energy[i] + Pooling_energy[i]
            if sum_energy == 0:
                CU_energy[i], Edram_buffer_energy[i] = 0, 0
                Bus_energy[i], Shift_and_add_energy[i] = 0, 0
                Or_energy[i], Activation_energy[i] = 0, 0
                Pooling_energy[i] = 0
            else:
                CU_energy[i], Edram_buffer_energy[i] = CU_energy[i]*100/sum_energy, Edram_buffer_energy[i]*100/sum_energy
                Bus_energy[i], Shift_and_add_energy[i] = Bus_energy[i]*100/sum_energy, Shift_and_add_energy[i]*100/sum_energy
                Or_energy[i], Activation_energy[i] = Or_energy[i]*100/sum_energy, Activation_energy[i]*100/sum_energy
                Pooling_energy[i] = Pooling_energy[i]*100/sum_energy

        plt.bar(idx, CU_energy,  width=0.8)
        energy_sum = CU_energy
        plt.bar(idx, Edram_buffer_energy, bottom=energy_sum,  width=0.8)
        energy_sum += Edram_buffer_energy
        plt.bar(idx, Bus_energy, bottom=energy_sum,  width=0.8)
        energy_sum += Bus_energy
        plt.bar(idx, Shift_and_add_energy, bottom=energy_sum,  width=0.8)
        energy_sum += Shift_and_add_energy
        plt.bar(idx, Or_energy, bottom=energy_sum,  width=0.8)
        energy_sum += Or_energy
        plt.bar(idx, Activation_energy, bottom=energy_sum,  width=0.8)
        energy_sum += Activation_energy
        plt.bar(idx, Pooling_energy, bottom=energy_sum,  width=0.8)
        plt.xlabel('PE index')
        plt.ylabel('Energy (%)')
        plt.legend(["CU_energy", "Edram_buffer_energy", "Bus_energy", "Shift_and_add_energy", "Or_energy", "Activation_energy", "Pooling_energy"])
        plt.savefig('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/Energy_breakdown_PE_percentage.png')
        plt.clf()

    def buffer_analysis(self):
        max_buffer_need = 0
        for i in range(len(self.PE_array)):
            max_buffer_need = max(self.PE_array[i].edram_buffer.maximal_usage * self.input_bit/8/1000, max_buffer_need)
            max_buffer_need = max(self.PE_array[i].edram_buffer_i.maximal_usage * self.input_bit/8/1000, max_buffer_need)
        ### Utilization
        with open('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/OnchipBuffer.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in range(self.cycle_ctr // self.buffer_record_freq):
                c = [(row+1)*self.buffer_record_freq] # cycle
                for i in range(len(self.PE_array)):
                    c.append(self.buffer_size[i][row])
                writer.writerow(c)
        self.buffer_size = np.array(self.buffer_size)
        for i in range(len(self.PE_array)):
            plt.plot(range(self.buffer_record_freq, self.cycle_ctr, self.buffer_record_freq), self.buffer_size[i] * self.input_bit/8/1000, label="PE"+str(i)) #, c=self.color[i])
        plt.title(self.mapping_str+", "+self.scheduling_str)
        plt.xlabel('Cycle')
        plt.ylabel('Buffer size(KB)')
        plt.legend(loc='best', prop={'size': 6})
        plt.savefig('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/Buffer_utilization.png')
        plt.clf()

        ### Maximal usage
        for i in range(len(self.PE_array)):
            plt.bar(i, self.PE_array[i].edram_buffer.maximal_usage * self.input_bit/8/1000, color='b', width=0.8)
        plt.legend(["Maximal usage"])
        plt.title(self.mapping_str+", "+self.scheduling_str)
        plt.xlabel('PE index')
        plt.ylabel('Buffer Size(KB)')
        plt.ylim((0, max_buffer_need))
        plt.savefig('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/Buffer_maximal_usage.png')
        plt.clf()

        ### Utilization (ideal)
        with open('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/OnchipBuffer_i.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in range(self.cycle_ctr // self.buffer_record_freq):
                c = [(row+1)*self.buffer_record_freq]
                for i in range(len(self.PE_array)):
                    c.append(self.buffer_size_i[i][row])
                writer.writerow(c)
        self.buffer_size_i = np.array(self.buffer_size_i)
        for i in range(len(self.PE_array)):
            plt.plot(range(self.buffer_record_freq, self.cycle_ctr, self.buffer_record_freq), self.buffer_size_i[i] * self.input_bit/8/1000, label="PE"+str(i)) #, c=self.color[i])
            #plt.plot(range(1, self.cycle_ctr+1), self.buffer_size_i[i], label="PE"+str(i)) #, c=self.color[i])
        
        plt.title(self.mapping_str+", "+self.scheduling_str)
        plt.xlabel('Cycle')
        plt.ylabel('Buffer size(KB)')
        plt.legend(loc='best', prop={'size': 6})
        plt.savefig('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/Buffer_utilization_i.png')
        plt.clf()

        ### Maximal usage (ideal)
        for i in range(len(self.PE_array)):
            plt.bar(i, self.PE_array[i].edram_buffer_i.maximal_usage * self.input_bit/8/1000, color='b', width=0.8)
        plt.legend(["Maximal usage"])
        plt.title(self.mapping_str+", "+self.scheduling_str)
        plt.xlabel('PE index')
        plt.ylabel('Buffer Size(KB)')
        plt.ylim((0, max_buffer_need))
        plt.savefig('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/Buffer_maximal_usage_i.png')
        plt.clf()

    def energy_utilization(self):
        with open('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/Energy.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in range(0, self.cycle_ctr, self.freq):
                writer.writerow([row+1, self.energy_utilization[row]])
        plt.bar(range(1, self.cycle_ctr+1), self.energy_utilization)
        plt.title(self.mapping_str+", "+self.scheduling_str)
        plt.ylabel('Energy (nJ)')
        plt.xlabel('Cycle')
        plt.ylim([0, 20])
        plt.savefig('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/energy_utilization.png')
        plt.clf()

    def pe_utilization(self):
        with open('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/PE_utilization.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in range(0, len(self.pe_state_for_plot[0]), self.freq):
                writer.writerow([self.pe_state_for_plot[0][row], self.pe_state_for_plot[1][row]])
        plt.figure(figsize=(self.cycle_ctr/20, len(self.PE_array)/2))
        plt.scatter(self.pe_state_for_plot[0], self.pe_state_for_plot[1], s=2, c='blue')
        plt.title(self.mapping_str+", "+self.scheduling_str)
        plt.xlabel('Cycle')
        plt.ylabel('PE index')
        plt.ylim(-1, len(self.PE_array))
        plt.xlim(0, self.cycle_ctr)
        plt.yticks(range(0, len(self.PE_array)))
        plt.xticks(range(0, self.cycle_ctr, 10))
        plt.savefig('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/PE_utilization.png', bbox_inches='tight')
        plt.clf()

    def cu_utilization(self):
        with open('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/CU_utilization.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in range(0, len(self.cu_state_for_plot[0]), self.freq):
                writer.writerow([self.cu_state_for_plot[0][row], self.cu_state_for_plot[1][row]])
        plt.figure(figsize=(self.cycle_ctr/20, len(self.PE_array)*self.hd_info.CU_num/10))
        plt.scatter(self.cu_state_for_plot[0], self.cu_state_for_plot[1], s=2, c='blue')
        plt.title(self.mapping_str+", "+self.scheduling_str)
        plt.xlabel('Cycle')
        plt.ylabel('CU index')
        plt.ylim(-1, len(self.PE_array)*self.hd_info.CU_num)
        plt.xlim(0, self.cycle_ctr)
        plt.yticks(range(0, len(self.PE_array)*self.hd_info.CU_num, self.hd_info.CU_num))
        plt.xticks(range(0, self.cycle_ctr, 10))
        plt.savefig('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/CU_utilization.png', bbox_inches='tight')
        plt.clf()

    def crossbar_utilization(self):
        with open('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/XB_utilization.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in range(0, len(self.xb_state_for_plot[0]), self.freq):
                writer.writerow([self.xb_state_for_plot[0][row], self.xb_state_for_plot[1][row]])
        '''
        plt.figure(figsize=(self.cycle_ctr/20, len(self.PE_array)*self.hd_info.CU_num*self.hd_info.Xbar_num/10))
        plt.scatter(self.xb_state_for_plot[0], self.xb_state_for_plot[1], s=2, c='blue')
        plt.title(self.mapping_str+", "+self.scheduling_str)
        plt.xlabel('Cycle')
        plt.ylabel('XB index')
        plt.ylim(-1, len(self.PE_array)*self.hd_info.CU_num*self.hd_info.Xbar_num)
        plt.xlim(0, self.cycle_ctr)
        plt.yticks(range(0, len(self.PE_array)*self.hd_info.CU_num*self.hd_info.Xbar_num, self.hd_info.Xbar_num))
        plt.xticks(range(0, self.cycle_ctr, 10))
        plt.savefig('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/XB_utilization.png', bbox_inches='tight')
        plt.clf()
        '''

    def non_pipeline_stage(self):
        with open('./statistics/'+self.mapping_str+'/'+self.scheduling_str+'/stage.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in range(self.cycle_ctr):
                writer.writerow([row+1, self.pipeline_stage_record[row]])
