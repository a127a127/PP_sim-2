from PE import PE

from EventMetaData import EventMetaData
from FetchEvent import FetchEvent
from Interconnect import Interconnect
from Packet import Packet

import csv
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

import time
import gc

class Controller(object):
    def __init__(self, model_config, hw_config, ordergenerator, trace, mapping_str, scheduling_str, replacement, path):
        self.congestion =  True
        self.model_config = model_config
        self.hw_config = hw_config
        self.ordergenerator = ordergenerator
        self.trace = trace
        self.mapping_str = mapping_str
        self.scheduling_str = scheduling_str
        self.interconnect = Interconnect(self.hw_config)
        if self.scheduling_str == "Pipeline":
            self.isPipeLine = True
        elif self.scheduling_str == "Non-pipeline":
            self.isPipeLine = False
        if replacement == "Ideal":
            self.ideal_replacement = True
        elif replacement == "LRU":
            self.ideal_replacement = False

        self.path = path
        
        self.mp_info = ordergenerator.mp_info
        self.Computation_order = self.ordergenerator.Computation_order
        print("Computation order length:", len(self.Computation_order))
        
        self.input_bit = self.ordergenerator.model_info.input_bit

        self.PE_array = dict()
        for rty_idx in range(self.hw_config.Router_num_y):
            for rtx_idx in range(self.hw_config.Router_num_x):
                for pey_idx in range(self.hw_config.PE_num_y):
                    for pex_idx in range(self.hw_config.PE_num_x):
                        pe_pos = (rty_idx, rtx_idx, pey_idx, pex_idx)
                        pe = PE(self.hw_config, self.model_config.input_bit, pe_pos)
                        self.PE_array[pe_pos] = pe

        self.fetch_array = []
        
        self.Total_energy_interconnect = 0

        # Event queue
        self.data_transfer_erp = []

        # Event queue's PE index
        self.edram_rd_pe_idx     = set()
        self.cu_operation_pe_idx = set()
        self.pe_saa_pe_idx       = set()
        self.activation_pe_idx   = set()
        self.edram_wr_pe_idx     = set()
        self.pooling_pe_idx      = set()

        self.Trigger = dict()

        # Utilization
        self.pe_state_for_plot = [0] # [{PE1, PE2}, {PE1, PE2}, {PE1}, {PE1}, ...]
        self.cu_state_for_plot = [0] # [{CU1, CU2}, {CU1, CU2}, ...]
        self.layer_state_for_plot = [0] # [{layer0, layer1}, {layer0}, {layer0, layer1}...]
        self.layer_finish_cycle = []
        for nlayer in range(len(model_config.layer_list)):
            self.layer_finish_cycle.append(0)

        # pe performance breakdown
        # self.pe_state_cu_busy = [0] # [{PE1, PE2}, {PE1, PE2}, {PE1}, {PE1}, ...]
        # self.pe_state_wait_transfer = [0]

        # CU performance breakdonw new
        self.data_table = []
        model_info = self.ordergenerator.model_info
        for fm in range(model_info.layer_length+1):
            self.data_table.append([])
            for data_pos in range(model_info.input_h[fm] * model_info.input_w[fm] * model_info.input_c[fm]):
                self.data_table[fm].append([0,0,0]) # start time, end time, end transfer time
        
        # Pipeline control
        if not self.isPipeLine: # Non_pipeline
            self.pipeline_layer_stage = 0
            self.events_each_layer = []
            for layer in range(self.ordergenerator.model_info.layer_length):
                self.events_each_layer.append(0)
            for e in self.Computation_order:
                self.events_each_layer[e.nlayer] += 1
            self.this_layer_event_ctr = 0
            self.this_layer_cycle_ctr = 0
            self.cycles_each_layer = []
            print("events_each_layer:", self.events_each_layer)

            self.Non_pipeline_trigger = []

        self.busy_xb = 0

        self.run()
        self.print_statistics_result()

    def run(self):
        print("estimation start")
        for event in self.Computation_order:
            if event.preceding_event_count == 0:
                # append edram read event
                pos = event.position_idx
                pe_idx = (pos[0], pos[1], pos[2], pos[3])
                pe = self.PE_array[pe_idx]
                cu_idx = pos[4]
                pe.edram_rd_ir_erp[cu_idx].append(event)

                self.edram_rd_pe_idx.add(pe) # 要檢查的PE idx
                if cu_idx not in pe.edram_rd_cu_idx:
                    pe.edram_rd_cu_idx.append(cu_idx) # 要檢查的CU idx

            if event.nlayer != 0:
                break
        
        self.t_edram , self.t_cuop , self.t_pesaa, self.t_act, self.t_wr, self.t_pool, self.t_transfer, self.t_fetch, self.t_trigger, self.t_inter = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        # self.t_buffer = 0

        start_time = time.time()
        
        self.cycle_ctr = 0
        self.done_event = 0

        while True:
            if self.cycle_ctr % 10000 == 0:
                if self.done_event == 0:
                    pass
                else:
                    print("Completed:", int(self.done_event/len(self.Computation_order) * 100), "%")
                    print("Model:", self.model_config.Model_type)
                    print("Mapping:", self.mapping_str , "Scheduling:", self.scheduling_str)
                    print("Cycle",self.cycle_ctr, "Done event:", self.done_event, "time per event", (time.time()-start_time)/self.done_event, "time per cycle", (time.time()-start_time)/self.cycle_ctr)
                    print("edram:", self.t_edram, "t_cuop", self.t_cuop, "pesaa", self.t_pesaa, "act", self.t_act, "wr", self.t_wr)
                    print("pooling:", self.t_pool, "transfer", self.t_transfer, "fetch", self.t_fetch)
                    print("trigger", self.t_trigger, "interconnect", self.t_inter)
                    print()
                    self.t_edram, self.t_cuop, self.t_pesaa, self.t_act, self.t_wr = 0, 0, 0, 0, 0
                    self.t_pool, self.t_transfer, self.t_fetch = 0, 0, 0
                    self.t_trigger, self.t_inter = 0, 0
                    #self.t_buffer =  0
            
            if self.cycle_ctr % 1000000 == 0:
                if self.done_event == 0:
                    pass
                else:
                    collected = gc.collect()
                    print("Garbage collector: collected", collected, "objects.")

            self.cycle_ctr += 1

            if len(self.pe_state_for_plot) == self.cycle_ctr:
                self.pe_state_for_plot.append(set())
            if len(self.layer_state_for_plot) == self.cycle_ctr:
                self.layer_state_for_plot.append(set())
            # if len(self.pe_state_cu_busy) == self.cycle_ctr:
            #     self.pe_state_cu_busy.append(set())
            # if len(self.pe_state_wait_transfer) == self.cycle_ctr:
            #     self.pe_state_wait_transfer.append(set())

            ## Pipeline stage control ###
            if not self.isPipeLine: # Non_pipeline
                self.this_layer_cycle_ctr += 1
                if self.this_layer_event_ctr == self.events_each_layer[self.pipeline_layer_stage]:
                    self.pipeline_layer_stage += 1
                    print("\n------Layer", self.pipeline_layer_stage)
                    self.cycles_each_layer.append(self.this_layer_cycle_ctr)
                    self.this_layer_cycle_ctr = 0
                    self.this_layer_event_ctr = 0

                    # cu_dict = dict() # {PE0:{cu0, cu1}, PE1:{cu0, cu1}} # cu performance breakdown
                    pe_set = set() # {PE0, PE1, ...}
                    for trigger in self.Non_pipeline_trigger:
                        pe = trigger[0]
                        event = trigger[1]
                        if event.event_type == "edram_rd_ir":
                            data_list = trigger[2]
                            for data in data_list:
                                pe.edram_buffer.put(data, data)
                            cu_idx = event.position_idx[4]
                            pe.edram_rd_ir_erp[cu_idx].append(event)
                            self.edram_rd_pe_idx.add(pe)
                            if cu_idx not in pe.edram_rd_cu_idx:
                                pe.edram_rd_cu_idx.append(cu_idx)
                            
                            # cu performance breakdown
                            # if pe not in cu_dict:
                            #     cu_dict[pe] = {cu_idx}
                            # else:
                            #     cu_dict[pe].add(cu_idx)
                            
                            # pe performance breakdown
                            # pe_set.add(pe)
                            
                        elif event.event_type == "edram_rd":
                            if len(trigger) == 3: # transfer data此時寫入buffer 
                                data_list = trigger[2]
                                for data in data_list:
                                    pe.edram_buffer.put(data, data)
                            pe.edram_rd_erp.append(event)
                            self.edram_rd_pe_idx.add(pe)

                            # pe performance breakdown
                            # pe_set.add(pe)

                        else:
                            print("error event type:", event.event_type)
                    
                    # cu performance breakdown
                    # for pe in cu_dict:
                    #     for cu_idx in cu_dict[pe]:
                    #         pe.cu_wait_transfer[cu_idx] += (self.cycle_ctr - self.transfer_start_cycle)

                    # pe performance breakdown
                    # for pe in pe_set:
                    #     pe.pe_wait_transfer += (self.cycle_ctr - self.transfer_start_cycle)

                    self.Non_pipeline_trigger = []

            if self.trace:
                print("cycle:", self.cycle_ctr)
            self.Trigger_event()
            self.event_edram_rd()
            self.event_cu_op()
            self.event_pe_saa()
            self.event_act()
            self.event_edram_wr()
            self.event_pooling()
            self.event_transfer()
            self.fetch()

            tt = time.time()
            for s in range(self.hw_config.interconnect_step_num):
                self.interconnect_fn()
            self.t_inter += time.time() - tt

            # self.record_buffer_util()
            
            ### Finish
            if self.done_event == len(self.Computation_order):
                break

    def Trigger_event(self):
        tt = time.time()
        if self.cycle_ctr in self.Trigger:
            for trigger in self.Trigger[self.cycle_ctr]:
                pe = trigger[0]
                event = trigger[1]
                
                if event.event_type == "edram_rd_ir":
                    # transfer_data = trigger[2]
                    # for data in transfer_data:
                    #     pe.edram_buffer.put(data, data)
                    cu_idx = event.position_idx[4]
                    pe.edram_rd_ir_erp[cu_idx].appendleft(event)
                    if not pe.cu_state[cu_idx]: # state == False
                        self.edram_rd_pe_idx.add(pe)
                        if cu_idx not in pe.edram_rd_cu_idx:
                            pe.edram_rd_cu_idx.appendleft(cu_idx)
                    
                elif event.event_type == "cu_operation":
                    cu_idx = event.position_idx[4]
                    pe.cu_operation_erp[cu_idx].appendleft(event)
                    self.cu_operation_pe_idx.add(pe)
                    pe.cu_operation_cu_idx.appendleft(cu_idx)
                
                elif event.event_type == "pe_saa":
                    if len(trigger) == 3: # 讓此CU可以做其他event
                        cu_idx = trigger[2]
                        if pe.edram_rd_ir_erp[cu_idx]:
                            if cu_idx not in pe.edram_rd_cu_idx:
                                pe.edram_rd_cu_idx.append(cu_idx)
                            self.edram_rd_pe_idx.add(pe)
                        pe.cu_state[cu_idx] = False
                    pe.pe_saa_erp.append(event)
                    self.pe_saa_pe_idx.add(pe)
                    
                elif event.event_type == "edram_wr":
                    pe.edram_wr_erp.append(event)
                    self.edram_wr_pe_idx.add(pe)
                
                elif event.event_type == "activation":
                    pe.activation_erp.append(event)
                    self.activation_pe_idx.add(pe)
                
                elif event.event_type == "data_transfer":
                    self.data_transfer_erp.append(event)
                
                elif event.event_type == "edram_rd":
                    if len(trigger) == 3: # transfer data此時寫入buffer 
                        transfer_data = trigger[2]
                        for data in transfer_data:
                            pe.edram_buffer.put(data, data)
                    pe.edram_rd_erp.appendleft(event)
                    self.edram_rd_pe_idx.add(pe)
                
                elif event.event_type == "pooling":
                    pe.pooling_erp.append(event)
                    self.pooling_pe_idx.add(pe)

                else:
                    print("error event type:", event.event_type)

            del self.Trigger[self.cycle_ctr]
        
        self.t_trigger += time.time() - tt

    def event_edram_rd(self):
        tt = time.time()
        check_pe_idx = set()
        for pe in self.edram_rd_pe_idx:
            if pe.edram_rd_erp: # 非CU的edram read
                event = pe.edram_rd_erp.popleft()
                if pe.edram_rd_erp or pe.edram_rd_cu_idx:
                    check_pe_idx.add(pe)
            else:
                cu_idx = pe.edram_rd_cu_idx.popleft()
                event = pe.edram_rd_ir_erp[cu_idx].popleft()
                if pe.edram_rd_cu_idx:
                    check_pe_idx.add(pe)
            
            edram_rd_data = event.inputs
                    
            fetch_data = []
            if event.event_type == "edram_rd_ir":
                # data in buffer?
                for data in edram_rd_data:
                    if not pe.edram_buffer.get(data):
                        fetch_data.append(data)
                        pe.edram_buffer.miss += 1

            if fetch_data:
                # fetch data
                if self.trace:
                    print("\tfetch edram_rd_ir event_idx:", self.Computation_order.index(event))
                des_pe = pe
                self.fetch_array.append(FetchEvent(event, des_pe, fetch_data))

            else: # do edram rd
                if self.trace:
                    if event.event_type == "edram_rd_ir":
                        print("\tdo edram_rd_ir event_idx:", self.Computation_order.index(event),", layer", event.nlayer)
                    elif event.event_type == "edram_rd":
                        print("\tdo edram_rd event_idx:", self.Computation_order.index(event),", layer", event.nlayer)
                    
                self.done_event += 1
                if not self.isPipeLine:
                    self.this_layer_event_ctr += 1
                    
                # Energy
                num_data = len(edram_rd_data)
                if event.event_type == "edram_rd_ir":
                    pe.Edram_buffer_energy += self.hw_config.Energy_edram_buffer * self.input_bit * num_data # read
                    pe.Bus_energy += self.hw_config.Energy_bus * self.input_bit * num_data # bus
                    pe.CU_IR_energy += self.hw_config.Energy_ir_in_cu * self.input_bit * num_data # write
                else: # edram_rd
                    pe.Edram_buffer_energy += self.hw_config.Energy_edram_buffer * self.input_bit * num_data # read
                    pe.Bus_energy += self.hw_config.Energy_bus * self.input_bit * num_data # bus
                
                # Trigger
                pro_event_idx = event.proceeding_event[0] # only one pro event
                pro_event = self.Computation_order[pro_event_idx]
                pro_event.current_number_of_preceding_event += 1

                finish_cycle = self.cycle_ctr + 1
                
                if finish_cycle not in self.Trigger:
                    self.Trigger[finish_cycle] = [[pe, pro_event]]
                else:
                    self.Trigger[finish_cycle].append([pe, pro_event])
                
                # PE util
                self.pe_state_for_plot[self.cycle_ctr].add(pe)

                # layer
                self.layer_state_for_plot[self.cycle_ctr].add(event.nlayer)
                if self.cycle_ctr > self.layer_finish_cycle[event.nlayer]:
                    self.layer_finish_cycle[event.nlayer] = self.cycle_ctr

                # CU performance breakdwon new
                if event.event_type == "edram_rd_ir":
                    busy_start_time = self.cycle_ctr
                    busy_end_time = pe.cu_busy_end_time[cu_idx]
                    idle = [busy_end_time, busy_start_time]
    
                    model_info = self.ordergenerator.model_info
                    layer = event.nlayer
                    if layer == 0:
                        prelayer = 0
                        compute_n_overlap = 0
                        p_transfer = busy_start_time - busy_end_time
                        pe.cu_performance_breakdown[cu_idx][0] += prelayer
                        pe.cu_performance_breakdown[cu_idx][1] += compute_n_overlap
                        pe.cu_performance_breakdown[cu_idx][2] += p_transfer
                    else:
                        data_list = []
                        for data in edram_rd_data:
                            h, w, c = data[1], data[2], data[3]
                            pos = w + h * model_info.input_w[layer] + c * model_info.input_w[layer] * model_info.input_h[layer]
                            data_start_compute_time = self.data_table[layer][pos][0]
                            data_end_compute_time = self.data_table[layer][pos][1]
                            data_list.append([data_start_compute_time, data_end_compute_time])
                        
                        prelayer, compute_n_overlap, p_transfer = self.overlap_a(idle, data_list)
                        pe.cu_performance_breakdown[cu_idx][0] += prelayer
                        pe.cu_performance_breakdown[cu_idx][1] += compute_n_overlap
                        pe.cu_performance_breakdown[cu_idx][2] += p_transfer

                # free mem
                # event_idx = self.Computation_order.index(event)
                # self.Computation_order[event_idx] = 0
                
        self.edram_rd_pe_idx = check_pe_idx

        self.t_edram += time.time() - tt
    
    def overlap_a(self, line, data_list):
        flag  = True
        for d in data_list:
            s, e = d[0], d[1]

            for i in range(len(line)):
                if s <= line[i]:
                    I1 = i
                    break
            for i in range(len(line)):
                if e <= line[i]:
                    I2 = i
                    break

            if flag: # start from idle
                if I1 == 0:
                    flag = False
                    if I2 % 2 == 0:
                        del line[I1+1 : I2]
                    else:
                        line.insert(I2, e)
                        del line[I1+1 : I2]
                else:
                    if I1 % 2 == 1 and I2 % 2 == 1:
                        line.insert(I1, s)
                        line.insert(I2+1, e)
                        del line[I1+1: I2+1]
                    elif I1 % 2 == 1 and I2 % 2 == 0:
                        line.insert(I1, s)
                        del line[I1+1 : I2+1]
                    elif I1 % 2 == 0 and I2 % 2 == 0:
                        del line[I1 : I2]
                    elif I1 % 2 == 0 and I2 % 2 == 1:
                        line.insert(I2, e)
                        del line[I1 : I2]
            else: # start from busy
                if I1 % 2 == 1 and I2 % 2 == 1:
                    del line[I1 : I2]
                elif I1 % 2 == 1 and I2 % 2 == 0:
                    line.insert(I2, e)
                    del line[I1 : I2]
                elif I1 % 2 == 0 and I2 % 2 == 0:
                    line.insert(I1, s)
                    line.insert(I2+1, e)
                    del line[I1+1 : I2+1]
                elif I1 % 2 == 0 and I2 % 2 == 1:
                    line.insert(I1, s)
                    del line[I1+1 : I2+1]

        prelayer = 0
        compute_n_overlap = 0
        transfer = 0
        if flag:
            prelayer += line[1] - line[0]
            line = line[1:]
        
        for i in range(0, len(line) - 1, 2):
            compute_n_overlap += line[i+1] - line[i]

        transfer += line[-1] - line[0] - compute_n_overlap
        return prelayer, compute_n_overlap, transfer
       
    def event_cu_op(self):
        tt = time.time()
        for pe in self.cu_operation_pe_idx:
            cu_idx = pe.cu_operation_cu_idx.popleft()
            pe.cu_state[cu_idx] = True
            event = pe.cu_operation_erp[cu_idx].popleft()
            
            if self.trace:
                print("\tcu_operation event_idx:", self.Computation_order.index(event))
            
            self.done_event += 1
            if not self.isPipeLine:
                self.this_layer_event_ctr += 1
            
            # Energy
            ou_num_dict = event.inputs[1]
            for xb_idx in ou_num_dict:
                ou_num = ou_num_dict[xb_idx]
                pe.CU_dac_energy += self.hw_config.Energy_ou_dac * ou_num
                pe.CU_crossbar_energy += self.hw_config.Energy_ou_crossbar * ou_num
                pe.CU_adc_energy += self.hw_config.Energy_ou_adc * ou_num
                pe.CU_shift_and_add_energy += self.hw_config.Energy_ou_ssa * ou_num
                
                pe.CU_IR_energy += self.hw_config.Energy_ir_in_cu * ou_num * self.hw_config.OU_h 
                pe.CU_OR_energy += self.hw_config.Energy_or_in_cu * ou_num * self.hw_config.OU_w * self.hw_config.ADC_resolution

                self.busy_xb += ou_num

            # Trigger
            pro_event_idx = event.proceeding_event[0] # only one pro event
            pro_event = self.Computation_order[pro_event_idx]
            
            finish_cycle = self.cycle_ctr + event.inputs[0] + 2 # +2: pipeline last two stage
            
            if finish_cycle not in self.Trigger:
                self.Trigger[finish_cycle] = [[pe, pro_event, cu_idx]]
            else:
                self.Trigger[finish_cycle].append([pe, pro_event, cu_idx])
            
            # PE util
            for cycle in range(len(self.pe_state_for_plot), finish_cycle):
                self.pe_state_for_plot.append(set())
            for cycle in range(self.cycle_ctr, finish_cycle):
                self.pe_state_for_plot[cycle].add(pe)
            
            # CU util
            # for cycle in range(len(self.cu_state_for_plot), finish_cycle):
            #     self.cu_state_for_plot.append(set())
            # for cycle in range(self.cycle_ctr, finish_cycle):
            #     cu_plot_idx = pe.plot_idx * self.hw_config.CU_num + cu_idx
            #     self.cu_state_for_plot[cycle].add(cu_plot_idx)

            # layer
            for cycle in range(len(self.layer_state_for_plot), finish_cycle):
                self.layer_state_for_plot.append(set())
            for cycle in range(self.cycle_ctr, finish_cycle):
                self.layer_state_for_plot[cycle].add(event.nlayer)
            if finish_cycle > self.layer_finish_cycle[event.nlayer]:
                self.layer_finish_cycle[event.nlayer] = finish_cycle

            # CU performance breakdown new
            for pos in event.outputs:
                start_time = self.cycle_ctr
                self.data_table[event.nlayer+1][pos][0] = start_time
            pe.cu_busy_end_time[cu_idx] = finish_cycle
            
            # cu performance breakdown
            # pe.cu_busy[cu_idx] += event.inputs + 2
            # pe.cu_finish_cycle[cu_idx] = max(finish_cycle, pe.cu_finish_cycle[cu_idx])

            # pe performance breakdown
            # for cycle in range(len(self.pe_state_cu_busy), finish_cycle):
            #     self.pe_state_cu_busy.append(set())
            # for cycle in range(self.cycle_ctr, finish_cycle):
            #     self.pe_state_cu_busy[cycle].add(pe)
            
            # free mem
            # event_idx = self.Computation_order.index(event)
            # self.Computation_order[event_idx] = 0
                
        self.cu_operation_pe_idx = set()

        self.t_cuop += time.time() - tt

    def event_pe_saa(self):
        tt = time.time()
        check_pe_idx = set()
        for pe in self.pe_saa_pe_idx:
            event = pe.pe_saa_erp.popleft()
            if self.trace:
                print("\tdo pe_saa event_idx:", self.Computation_order.index(event))
            
            self.done_event += 1
            if not self.isPipeLine:
                self.this_layer_event_ctr += 1
            
            # Energy
            saa_amount = event.inputs
            pe.Or_energy += self.hw_config.Energy_or * self.input_bit * saa_amount
            pe.Bus_energy += self.hw_config.Energy_bus * self.input_bit * saa_amount
            pe.PE_shift_and_add_energy += self.hw_config.Energy_shift_and_add * saa_amount
            pe.Bus_energy += self.hw_config.Energy_bus * self.input_bit * saa_amount
            pe.Or_energy += self.hw_config.Energy_or * self.input_bit * saa_amount

            # Trigger
            for proceeding_index in event.proceeding_event:
                pro_event = self.Computation_order[proceeding_index]
                pro_event.current_number_of_preceding_event += 1
                if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                    finish_cycle = self.cycle_ctr + 1
                    if finish_cycle not in self.Trigger:
                        self.Trigger[finish_cycle] = [[pe, pro_event]]
                    else:
                        self.Trigger[finish_cycle].append([pe, pro_event])
            
            # PE util
            self.pe_state_for_plot[self.cycle_ctr].add(pe)

            # layer
            self.layer_state_for_plot[self.cycle_ctr].add(event.nlayer)
            if self.cycle_ctr > self.layer_finish_cycle[event.nlayer]:
                self.layer_finish_cycle[event.nlayer] = self.cycle_ctr

            if pe.pe_saa_erp:
                check_pe_idx.add(pe)
            
            # free mem
            # event_idx = self.Computation_order.index(event)
            # self.Computation_order[event_idx] = 0
        
        self.pe_saa_pe_idx = check_pe_idx
        self.t_pesaa += time.time() - tt
              
    def event_act(self):
        tt = time.time()
        check_pe_idx = set()
        for pe in self.activation_pe_idx:
            event = pe.activation_erp.popleft()
            if self.trace:
                print("\tdo activation event_idx:", self.Computation_order.index(event))
            
            self.done_event += 1
            if not self.isPipeLine:
                self.this_layer_event_ctr += 1
            
            # Energy
            act_amount = event.inputs
            pe.Or_energy += self.hw_config.Energy_or * self.input_bit * act_amount
            pe.Bus_energy += self.hw_config.Energy_bus * self.input_bit * act_amount
            pe.Activation_energy += self.hw_config.Energy_activation * act_amount

            # Trigger
            for proceeding_index in event.proceeding_event:
                pro_event = self.Computation_order[proceeding_index]
                pro_event.current_number_of_preceding_event += 1
                finish_cycle = self.cycle_ctr + 1
                if finish_cycle not in self.Trigger:
                    self.Trigger[finish_cycle] = [[pe, pro_event]]
                else:
                    self.Trigger[finish_cycle].append([pe, pro_event])
            
            # PE util
            self.pe_state_for_plot[self.cycle_ctr].add(pe)

            # layer
            self.layer_state_for_plot[self.cycle_ctr].add(event.nlayer)
            if self.cycle_ctr > self.layer_finish_cycle[event.nlayer]:
                self.layer_finish_cycle[event.nlayer] = self.cycle_ctr

            if pe.activation_erp:
                check_pe_idx.add(pe)
            
            # free mem
            # event_idx = self.Computation_order.index(event)
            # self.Computation_order[event_idx] = 0

        self.activation_pe_idx = check_pe_idx

        self.t_act += time.time() - tt

    def event_edram_wr(self):
        tt = time.time()
        check_pe_idx = set()
        for pe in self.edram_wr_pe_idx:
            event = pe.edram_wr_erp.popleft()
            if self.trace:
                print("\tdo edram_wr event_idx:", self.Computation_order.index(event))
            
            self.done_event += 1
            if not self.isPipeLine:
                self.this_layer_event_ctr += 1
            
            # Energy
            edram_write_data = event.outputs
            num_data = len(edram_write_data)
            pe.Bus_energy += self.hw_config.Energy_bus * self.input_bit * num_data
            pe.Edram_buffer_energy += self.hw_config.Energy_edram_buffer * self.input_bit * num_data

            # Write data
            for data in edram_write_data:
                pe.edram_buffer.put(data, data)

            #self.check_buffer_pe_set.add(pe)

            # Trigger
            for proceeding_index in event.proceeding_event:
                pro_event = self.Computation_order[proceeding_index]
                pro_event.current_number_of_preceding_event += 1
                if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                    if not self.isPipeLine and pro_event.nlayer != self.pipeline_layer_stage: # Non_pipeline
                        self.Non_pipeline_trigger.append([pe, pro_event, edram_write_data])
                    else:
                        finish_cycle = self.cycle_ctr + 1
                        if finish_cycle not in self.Trigger:
                            self.Trigger[finish_cycle] = [[pe, pro_event]]
                        else:
                            self.Trigger[finish_cycle].append([pe, pro_event])


            # pe performance breakdown
            # pe.pe_finish_cycle = self.cycle_ctr

            # PE util
            self.pe_state_for_plot[self.cycle_ctr].add(pe)

            # layer
            self.layer_state_for_plot[self.cycle_ctr].add(event.nlayer)
            if self.cycle_ctr > self.layer_finish_cycle[event.nlayer]:
                self.layer_finish_cycle[event.nlayer] = self.cycle_ctr

            if pe.edram_wr_erp:
                check_pe_idx.add(pe)

            # free mem
            # event_idx = self.Computation_order.index(event)
            # self.Computation_order[event_idx] = 0

        self.edram_wr_pe_idx = check_pe_idx
        
        self.t_wr += time.time() - tt

    def event_pooling(self):
        tt = time.time()
        check_pe_idx = set()
        for pe in self.pooling_pe_idx:
            event = pe.pooling_erp.popleft()
            if self.trace:
                print("\tdo pooling event_idx:", self.Computation_order.index(event))
            
            self.done_event += 1
            if not self.isPipeLine:
                self.this_layer_event_ctr += 1
            
            # Energy
            pooling_amount = event.inputs
            pe.Pooling_energy += self.hw_config.Energy_pooling * pooling_amount

            # Trigger
            for proceeding_index in event.proceeding_event:
                pro_event = self.Computation_order[proceeding_index]
                pro_event.current_number_of_preceding_event += 1
                if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                    finish_cycle = self.cycle_ctr + 1
                    if finish_cycle not in self.Trigger:
                        self.Trigger[finish_cycle] = [[pe, pro_event]]
                    else:
                        self.Trigger[finish_cycle].append([pe, pro_event])
            
            # PE util
            self.pe_state_for_plot[self.cycle_ctr].add(pe)

            # layer
            self.layer_state_for_plot[self.cycle_ctr].add(event.nlayer)
            if self.cycle_ctr > self.layer_finish_cycle[event.nlayer]:
                self.layer_finish_cycle[event.nlayer] = self.cycle_ctr

            if pe.pooling_erp:
                check_pe_idx.add(pe)
            
            # free mem
            # event_idx = self.Computation_order.index(event)
            # self.Computation_order[event_idx] = 0

        self.pooling_pe_idx = check_pe_idx

        self.t_pool += time.time() - tt

    def event_transfer(self):
        tt = time.time()
        for event in self.data_transfer_erp:
            if self.trace:
                print("\tdo data_transfer event idx:", self.Computation_order.index(event))

            self.done_event += 1
            if not self.isPipeLine:
                self.this_layer_event_ctr += 1

            transfer_data = event.outputs
            data_transfer_src = event.position_idx[0]
            data_transfer_des = event.position_idx[1]
            src_pe = self.PE_array[data_transfer_src]
            des_pe = self.PE_array[data_transfer_des]

            if data_transfer_src == data_transfer_des:
                # Trigger
                finish_cycle = self.cycle_ctr + 1
                for pro_event_idx in event.proceeding_event:
                    pro_event = self.Computation_order[pro_event_idx]
                    pro_event.current_number_of_preceding_event += 1
                    if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                        if not self.isPipeLine and pro_event.nlayer != self.pipeline_layer_stage: # Non_pipeline
                            self.Non_pipeline_trigger.append([des_pe, pro_event, transfer_data])
                            # self.transfer_start_cycle = self.cycle_ctr # cu performance breakdown
                        else:
                            if finish_cycle not in self.Trigger:
                                self.Trigger[finish_cycle] = [[des_pe, pro_event, transfer_data]]
                            else:
                                self.Trigger[finish_cycle].append([des_pe, pro_event, transfer_data])
                    else:
                        for data in transfer_data: # 沒有trigger event先放data
                            des_pe.edram_buffer.put(data, data)
            else:
                num_data = len(transfer_data)
                transfer_distance  = abs(data_transfer_des[1] - data_transfer_src[1])
                transfer_distance += abs(data_transfer_des[0] - data_transfer_src[0])

                # Energy
                self.Total_energy_interconnect += self.hw_config.Energy_router * self.input_bit * num_data * (transfer_distance + 1)
                self.Total_energy_interconnect += self.hw_config.Energy_link * self.input_bit * num_data * transfer_distance
                des_pe.Edram_buffer_energy += self.hw_config.Energy_edram_buffer * self.input_bit * num_data # write

                if self.congestion:
                    model_info = self.ordergenerator.model_info
                    layer = event.nlayer+1
                    for i in range(len(transfer_data)-1):
                        data = transfer_data[i]
                        packet = Packet(data_transfer_src, data_transfer_des, data, [])
                        self.interconnect.input_packet(packet)
                        # CU performance breakdown new
                        h, w, c = data[1], data[2], data[3]
                        pos = w + h * model_info.input_w[layer] + c * model_info.input_w[layer] * model_info.input_h[layer]
                        self.data_table[layer][pos][1] = self.cycle_ctr - 1 # end compute time

                    data = transfer_data[-1]
                    h, w, c = data[1], data[2], data[3]
                    packet = Packet(data_transfer_src, data_transfer_des, data, event.proceeding_event)
                    self.interconnect.input_packet(packet)
                    # self.transfer_start_cycle = self.cycle_ctr # cu performance breakdown

                    # CU performance breakdown new
                    pos = w + h * model_info.input_w[layer] + c * model_info.input_w[layer] * model_info.input_h[layer]
                    self.data_table[layer][pos][1] = self.cycle_ctr - 1  # end compute time

                else:
                    # Trigger
                    # pe_set = set() # pe performance breakdown
                    # TODO: CU performance breakdown new
                    finish_cycle = self.cycle_ctr + 1 + transfer_distance + 1
                    for pro_event_idx in event.proceeding_event:
                        pro_event = self.Computation_order[pro_event_idx]
                        pro_event.current_number_of_preceding_event += 1
                        if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                            if not self.isPipeLine and pro_event.nlayer != self.pipeline_layer_stage: # Non_pipeline
                                self.Non_pipeline_trigger.append([des_pe, pro_event, transfer_data])
                                # self.transfer_start_cycle = self.cycle_ctr # cu performance breakdown
                            else:
                                if finish_cycle not in self.Trigger:
                                    self.Trigger[finish_cycle] = [[des_pe, pro_event, transfer_data]]
                                else:
                                    self.Trigger[finish_cycle].append([des_pe, pro_event, transfer_data])
                                
                                # if pro_event.event_type == "edram_rd_ir":
                                #     # cu performance breakdown
                                #     cu_idx = pro_event.position_idx[4]
                                #     des_pe.cu_wait_transfer[cu_idx] += transfer_distance + 1

                                # pe performance breakdown
                                # pe_set.add(des_pe)
                        else:
                            for data in transfer_data: # 沒有trigger event先放data
                                des_pe.edram_buffer.put(data, data)
                
                    # pe performance breakdown
                    # for pe in pe_set:
                    #     for cycle in range(len(self.pe_state_wait_transfer), finish_cycle):
                    #         self.pe_state_wait_transfer.append(set())
                    #     for cycle in range(self.cycle_ctr+1, finish_cycle):
                    #         self.pe_state_wait_transfer[cycle].add(pe)

            # pe performance breakdown
            # src_pe.pe_finish_cycle = self.cycle_ctr - 1

            # free mem
            # event_idx = self.Computation_order.index(event)
            # self.Computation_order[event_idx] = 0

        self.data_transfer_erp = []

        self.t_transfer += time.time() - tt

    def fetch(self):
        tt = time.time()
        for fe in self.fetch_array:
            transfer_data = fe.data
            num_data = len(transfer_data)
            des_pe = fe.des_pe
            event = fe.event
            transfer_distance = des_pe.position[0]

            # Energy
            self.Total_energy_interconnect += self.hw_config.Energy_router * self.input_bit * num_data * (transfer_distance + 1)
            des_pe.Edram_buffer_energy += self.hw_config.Energy_edram_buffer * self.input_bit * num_data # write

            # Cycle
            finish_cycle = self.cycle_ctr + 1 + self.hw_config.Fetch_cycle + transfer_distance + 1 # TODO: fix/ affects CU performance breakdown
            
            # Trigger
            if finish_cycle not in self.Trigger:
                self.Trigger[finish_cycle] = [[des_pe, event]]
            else:
                self.Trigger[finish_cycle].append([des_pe, event])
            for data in transfer_data:
                des_pe.edram_buffer.put(data, data)
            
            
            # cu performance breakdown
            # pos_idx = event.position_idx
            # pe_pos = pos_idx[:4]
            # pe = self.PE_array[pe_pos]
            # if event.event_type == "edram_rd_ir":
            #     cu_idx = pos_idx[4]
            #     pe.cu_wait_transfer[cu_idx] += self.hw_config.Fetch_cycle + transfer_distance + 2

            # pe performance breakdown
            # for cycle in range(len(self.pe_state_wait_transfer), finish_cycle):
            #     self.pe_state_wait_transfer.append(set())
            # for cycle in range(self.cycle_ctr, finish_cycle):
            #     self.pe_state_wait_transfer[cycle].add(pe)

        self.fetch_array = []

        self.t_fetch += time.time() - tt

    def interconnect_fn(self):
        arrived = self.interconnect.step()
        model_info = self.ordergenerator.model_info
        for packet in arrived:
            des_pe_id = packet.destination
            des_pe = self.PE_array[des_pe_id]
            data = packet.data
            pro_event_list = packet.pro_event_list
            des_pe.edram_buffer.put(data, data)

            # CU performance breakdown
            layer, h, w, c = data[0], data[1], data[2], data[3]
            pos = w + h * model_info.input_w[layer+1] + c * model_info.input_w[layer+1] * model_info.input_h[layer+1]
            self.data_table[layer][pos][2] = self.cycle_ctr

            # Trigger
            for proceeding_index in pro_event_list:
                pro_event = self.Computation_order[proceeding_index]
                pro_event.current_number_of_preceding_event += 1
                if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                    if not self.isPipeLine and pro_event.nlayer != self.pipeline_layer_stage: # Non_pipeline
                        self.Non_pipeline_trigger.append([des_pe, pro_event, []])
                    else:
                        finish_cycle = self.cycle_ctr + 1
                        if finish_cycle not in self.Trigger:
                            self.Trigger[finish_cycle] = [[des_pe, pro_event]]
                        else:
                            self.Trigger[finish_cycle].append([des_pe, pro_event])

    def record_buffer_util(self):
        # time history
        tt = time.time()
        fre = 200
        if self.cycle_ctr % fre == 0:
            for pe in self.check_buffer_pe_set:
                pe.buffer_size_util[0].append(self.cycle_ctr)
                pe.buffer_size_util[1].append(len(pe.edram_buffer.buffer))
            self.check_buffer_pe_set = set()
        self.t_buffer += time.time() - tt

    def print_statistics_result(self):
        print("print_statistics_result")
        # self.color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        self.Edram_buffer_energy     = 0.
        self.Bus_energy              = 0.
        self.PE_shift_and_add_energy = 0.
        self.Or_energy               = 0.
        self.Activation_energy       = 0.
        self.Pooling_energy          = 0.

        self.CU_shift_and_add_energy = 0.
        self.CU_dac_energy           = 0.
        self.CU_adc_energy           = 0.
        self.CU_crossbar_energy      = 0.
        self.CU_IR_energy            = 0.
        self.CU_OR_energy            = 0.

        for pe_pos in self.PE_array:
            pe = self.PE_array[pe_pos]
            self.Edram_buffer_energy     += pe.Edram_buffer_energy
            self.Bus_energy              += pe.Bus_energy
            self.PE_shift_and_add_energy += pe.PE_shift_and_add_energy
            self.Or_energy               += pe.Or_energy
            self.Activation_energy       += pe.Activation_energy
            self.Pooling_energy          += pe.Pooling_energy
            self.CU_shift_and_add_energy += pe.CU_shift_and_add_energy
            self.CU_dac_energy           += pe.CU_dac_energy
            self.CU_adc_energy           += pe.CU_adc_energy
            self.CU_crossbar_energy      += pe.CU_crossbar_energy
            self.CU_IR_energy            += pe.CU_IR_energy
            self.CU_OR_energy            += pe.CU_OR_energy
        self.Total_energy = self.Edram_buffer_energy + self.Bus_energy + self.PE_shift_and_add_energy + \
                            self.Or_energy + self.Activation_energy + self.Pooling_energy + \
                            self.CU_shift_and_add_energy + self.CU_dac_energy + self.CU_adc_energy + \
                            self.CU_crossbar_energy + self.CU_IR_energy + self.CU_OR_energy + \
                            self.Total_energy_interconnect



        print("Total Cycles:", self.cycle_ctr)
        if not self.isPipeLine:
            print("Cycles each layer:", self.cycles_each_layer)
        print("Cycles time:", self.hw_config.cycle_time, "ns\n")
        print()

        self.output_result()
        print("Energy breakdown:")
        self.PE_energy_breakdown()
        print("output buffer utilization...")
        self.buffer_analysis()
        print("output pe utilization...")
        self.pe_utilization()
        #print("output cu utilization...")
        #self.cu_utilization()
        print("output layer utilization...")
        self.layer_utilization()
        print("output performance anaylsis...")
        # self.layer_behavior()
        self.cu_performance_breakdown()
        # self.pe_performance_breakdown()
        self.miss_rate()

    def output_result(self):
        with open(self.path+'/Result.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Event Total", len(self.Computation_order)])
            writer.writerow(["edram_rd_ir_ctr", self.ordergenerator.edram_rd_ir_ctr])
            writer.writerow(["cu_op_ctr", self.ordergenerator.cu_op_ctr])
            writer.writerow(["pe_saa_ctr", self.ordergenerator.pe_saa_ctr])
            writer.writerow(["edram_wr_ctr", self.ordergenerator.edram_wr_ctr])
            writer.writerow(["edram_rd_ctr", self.ordergenerator.edram_rd_ctr])
            writer.writerow(["pooling_ctr", self.ordergenerator.pooling_ctr])
            writer.writerow(["data_transfer_ctr", self.ordergenerator.data_transfer_ctr])
            
            writer.writerow([])
            writer.writerow(["Cycles", self.cycle_ctr])
            writer.writerow([])
            writer.writerow(["Total energy consumption(nJ)", self.Total_energy])
            writer.writerow([])
            writer.writerow(["", "Energy(nJ)"])
            writer.writerow(["Interconnect", self.Total_energy_interconnect])

            writer.writerow(["Edram Buffer", self.Edram_buffer_energy])
            writer.writerow(["Bus", self.Bus_energy])
            writer.writerow(["PE Shift and Add", self.PE_shift_and_add_energy])
            writer.writerow(["OR", self.Or_energy])
            writer.writerow(["Activation", self.Activation_energy])
            writer.writerow(["Pooling", self.Pooling_energy])

            writer.writerow(["CU Shift and Add", self.CU_shift_and_add_energy])
            writer.writerow(["DAC", self.CU_dac_energy])
            writer.writerow(["ADC", self.CU_adc_energy])
            writer.writerow(["Crossbar Array", self.CU_crossbar_energy])
            writer.writerow(["IR", self.CU_IR_energy])
            writer.writerow(["OR", self.CU_OR_energy])

            writer.writerow([])
            writer.writerow(["Busy XB", self.busy_xb])
            writer.writerow(["Avg", self.busy_xb/self.cycle_ctr])

            writer.writerow([])
            layer_used_xb_num = self.ordergenerator.mp_info.layer_used_xb_num
            writer.writerow([str(layer_used_xb_num)])


            writer.writerow([])
            fm_num  = self.ordergenerator.transfer_feature_map_data_num
            inter_num = self.ordergenerator.transfer_intermediate_data_num
            writer.writerow(["transfer data between PE"])
            writer.writerow(["feature map data", fm_num])
            writer.writerow(["intermediate data", inter_num])

    def PE_energy_breakdown(self):
        # PE breakdown
        print("PE energy breakdown")
        with open(self.path+'/PE_Energy_breakdown.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["", "Buffer", "Bus", "PE Shift and add", "OR", "Activation", "Pooling",
                             "CU Shift and add", "DAC", "ADC", "Crossbar", "IR", "OR"
                            ])
            for pe_pos in self.PE_array:
                pe = self.PE_array[pe_pos]
                if pe.Edram_buffer_energy != 0 or pe.Bus_energy != 0 or pe.PE_shift_and_add_energy != 0 or \
                    pe.Or_energy != 0 or pe.Activation_energy != 0 or pe.Pooling_energy != 0 or \
                    pe.CU_shift_and_add_energy != 0 or pe.CU_dac_energy != 0 or pe.CU_adc_energy != 0 or \
                    pe.CU_crossbar_energy != 0 or pe.CU_IR_energy != 0 or pe.CU_OR_energy != 0:
                        rty, rtx, pey, pex = pe_pos[0], pe_pos[1], pe_pos[2], pe_pos[3]
                        idx = pex + pey * self.hw_config.PE_num_x + rtx * self.hw_config.PE_num + rty * self.hw_config.PE_num * self.hw_config.Router_num_x
                        arr = ["PE"+str(idx)]
                        arr.append(pe.Edram_buffer_energy)
                        arr.append(pe.Bus_energy)
                        arr.append(pe.PE_shift_and_add_energy)
                        arr.append(pe.Or_energy)
                        arr.append(pe.Activation_energy)
                        arr.append(pe.Pooling_energy)

                        arr.append(pe.CU_shift_and_add_energy)
                        arr.append(pe.CU_dac_energy)
                        arr.append(pe.CU_adc_energy)
                        arr.append(pe.CU_crossbar_energy)
                        arr.append(pe.CU_IR_energy)
                        arr.append(pe.CU_OR_energy)
                        writer.writerow(arr)

    def buffer_analysis(self):
        # num轉KB
        for pe_pos in self.PE_array:
            pe = self.PE_array[pe_pos]
            pe.edram_buffer.maximal_usage *= self.input_bit/8/1000
            # for i in range(len(pe.buffer_size_util[1])):
            #     pe.buffer_size_util[1][i] *= self.input_bit/8/1000

        with open(self.path+'/Buffer_utilization.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["", "Size(KB)"])
            for pe_pos in self.PE_array:
                pe = self.PE_array[pe_pos]
                if pe.edram_buffer.maximal_usage == 0:
                    continue
                rty, rtx, pey, pex = pe_pos[0], pe_pos[1], pe_pos[2], pe_pos[3]
                idx = pex + pey * self.hw_config.PE_num_x + rtx * self.hw_config.PE_num + rty * self.hw_config.PE_num * self.hw_config.Router_num_x
                writer.writerow(["PE"+str(idx), pe.edram_buffer.maximal_usage])
        
        with open(self.path+'/Buffer_utilization2.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["", "tmp_data", "fm_data"])
            for pe_pos in self.PE_array:
                pe  = self.PE_array[pe_pos]
                buf = pe.edram_buffer.buffer
                tmp_data, fm_data = 0, 0
                for data in buf:
                    if len(data) == 5: # intermediate data
                        tmp_data += 1
                    elif len(data) == 4: # feature map data
                        fm_data += 1
                tmp_data *= self.input_bit/8/1000
                fm_data *= self.input_bit/8/1000
                if tmp_data == 0 and fm_data == 0:
                    continue
                rty, rtx, pey, pex = pe_pos[0], pe_pos[1], pe_pos[2], pe_pos[3]
                idx = pex + pey * self.hw_config.PE_num_x + rtx * self.hw_config.PE_num + rty * self.hw_config.PE_num * self.hw_config.Router_num_x
                writer.writerow(["PE"+str(idx), tmp_data, fm_data]) 

        # ### time history
        # with open(self.path+'/Buffer_time_history.csv', 'w', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     for pe in self.PE_array:
        #         util = pe.buffer_size_util
        #         if len(util[0]) == 0:
        #             continue
        #         writer.writerow(["PE"+str(self.PE_array.index(pe))])
        #         writer.writerow(util[0])
        #         writer.writerow(util[1])

    def pe_utilization(self):
        with open(self.path+'/PE_utilization.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for cycle in range(1, len(self.pe_state_for_plot)):
                if self.pe_state_for_plot[cycle]:
                    for pe in self.pe_state_for_plot[cycle]:
                        plot_idx = pe.plot_idx
                        writer.writerow([cycle, plot_idx])

    def cu_utilization(self):
        with open(self.path+'/CU_utilization.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for cycle in range(1, len(self.cu_state_for_plot)):
                if self.cu_state_for_plot[cycle]:
                    for cu_idx in self.cu_state_for_plot[cycle]:
                        writer.writerow([cycle, cu_idx])

    def layer_utilization(self):
        with open(self.path+'/Layer_utilization.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for cycle in range(1, len(self.layer_state_for_plot)):
                if self.layer_state_for_plot[cycle]:
                    for layer in self.layer_state_for_plot[cycle]:
                        writer.writerow([cycle, layer])

    def layer_behavior(self):
        self.layer_used_cycle = []
        for nlayer in range(len(self.model_config.layer_list)):
            self.layer_used_cycle.append(0)
        for cycle in range(1, len(self.layer_state_for_plot)):
            if self.layer_state_for_plot[cycle]:
                for layer in self.layer_state_for_plot[cycle]:
                    self.layer_used_cycle[layer] += 1
        
        with open(self.path+'/Layer_behavior.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["", "busy_time", "idle_time"])
            for nlayer in range(len(self.model_config.layer_list)):
                busy_time = self.layer_used_cycle[nlayer]
                idle_time = self.layer_finish_cycle[nlayer] - busy_time
                writer.writerow([nlayer, busy_time, idle_time])

    def cu_performance_breakdown(self):
        Plot = [] # [PE0, PE1, PE2, ...]
        p_total = 0
        o_total = 0
        t_total = 0
        for pe_n in range(len(self.PE_array)):
            Plot.append([])
        for pe_pos in self.PE_array:
            pe = self.PE_array[pe_pos]
            for cu_idx in range(self.hw_config.CU_num):
                bd = pe.cu_performance_breakdown[cu_idx]
                pre_layer, compute_overlap, transfer = bd[0], bd[1], bd[2]
                p_total += pre_layer
                o_total += compute_overlap
                t_total += transfer
                Plot[pe.plot_idx].append([pre_layer, compute_overlap, transfer])
        with open(self.path+'/CU_performance_breakdown.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["", "previous-layer", "idle for other", "idle for transfer"])
            for pe_idx in range(len(Plot)):
                cu_list = Plot[pe_idx]
                for cu_idx in range(len(cu_list)):
                    row = cu_list[cu_idx]
                    if row[0] != 0 or row[1] != 0 or row[2] !=0 :
                        row.insert(0, "PE"+str(pe_idx)+"_CU"+str(cu_idx))
                        writer.writerow(row)
            writer.writerow(["SUM", p_total, o_total, t_total])

        # Old version
        # for pe_pos in self.PE_array:
        #     pe = self.PE_array[pe_pos]
        #     for cu_idx in range(len(pe.cu_busy)):
        #         pe.cu_idle[cu_idx] = pe.cu_finish_cycle[cu_idx] - pe.cu_busy[cu_idx] - pe.cu_wait_transfer[cu_idx]
        
        # Plot = [] # [PE0, PE1, PE2, ...]
        # for pe_n in range(len(self.PE_array)):
        #     Plot.append([])
        # for pe_pos in self.PE_array:
        #     pe = self.PE_array[pe_pos]
        #     for cu_idx in range(self.hw_config.CU_num):
        #         cu_busy = pe.cu_busy[cu_idx]
        #         cu_wait_transfer = pe.cu_wait_transfer[cu_idx]
        #         cu_idle = pe.cu_idle[cu_idx]
        #         Plot[pe.plot_idx].append([cu_busy, cu_wait_transfer, cu_idle])

        # with open(self.path+'/CU_performance_breakdown.csv', 'w', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow(["", "busy", "wait_transfer", "idle"])
        #     for pe_idx in range(len(Plot)):
        #         cu_list = Plot[pe_idx]
        #         for cu_idx in range(len(cu_list)):
        #             row = cu_list[cu_idx]
        #             if row[0] != 0 or row[1] != 0 or row[2] !=0 :
        #                 row.insert(0, "PE"+str(pe_idx)+"_CU"+str(cu_idx))
        #                 writer.writerow(row)

    def pe_performance_breakdown(self):
        for cycle in range(1, len(self.pe_state_for_plot)):
            if self.pe_state_wait_transfer[cycle]:
                for pe in self.pe_state_wait_transfer[cycle]:
                    if pe not in self.pe_state_for_plot[cycle]:
                        pe.pe_wait_transfer += 1
            if self.pe_state_for_plot[cycle]:
                for pe in self.pe_state_for_plot[cycle]:
                    if pe in self.pe_state_cu_busy[cycle]:
                        pe.pe_busy_cu    += 1
                    else:
                        pe.pe_busy_other += 1
        
        for pe_pos in self.PE_array:
            pe = self.PE_array[pe_pos]
            pe.pe_idle = pe.pe_finish_cycle - pe.pe_busy_cu - pe.pe_busy_other - pe.pe_wait_transfer
        
        Plot = [] # [PE0, PE1, PE2, ...]
        for pe_n in range(len(self.PE_array)):
            Plot.append([])
        for pe_pos in self.PE_array:
            pe = self.PE_array[pe_pos]
            Plot[pe.plot_idx] = [pe.pe_busy_cu, pe.pe_busy_other, pe.pe_wait_transfer, pe.pe_idle]
        
        with open(self.path+'/PE_performance_breakdown.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["", "busy_cu", "busy_other", "wait_transfer", "idle"])
            for pe_idx in range(len(Plot)):
                row = Plot[pe_idx]
                if row[0] != 0 or row[1] != 0 or row[2] !=0 :
                    row.insert(0, "PE"+str(pe_idx))
                    writer.writerow(row)

    def miss_rate(self):
        with open(self.path+'/Miss_rate.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for pe_pos in self.PE_array:
                rty, rtx, pey, pex = pe_pos[0], pe_pos[1], pe_pos[2], pe_pos[3]
                idx = pex + pey * self.hw_config.PE_num_x + rtx * self.hw_config.PE_num + rty * self.hw_config.PE_num * self.hw_config.Router_num_x
                arr = ["PE"+str(idx)]

                pe = self.PE_array[pe_pos]
                miss = pe.edram_buffer.miss
                
                arr.append(miss)
                writer.writerow(arr)

