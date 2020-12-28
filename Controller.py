from PE import PE
from Interconnect import Interconnect
from Packet import Packet
import csv
import time

class Controller(object):
    def __init__(self, model_config, hw_config, ordergenerator, trace, mapping_str, scheduling_str, path, log):
        self.model_config = model_config
        self.hw_config = hw_config
        self.ordergenerator = ordergenerator
        self.trace = False
        self.mapping_str = mapping_str
        self.scheduling_str = scheduling_str
        self.path = path

        self.interconnect = Interconnect(self.hw_config)
        self.log = log
        
        if self.scheduling_str == "Pipeline":
            self.isPipeLine = True
        elif self.scheduling_str == "Non-pipeline":
            self.isPipeLine = False      

        self.record_PE = False
        self.record_layer = False
        
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

        self.fetch_dict = dict() # {cycle: list}
        
        self.Total_energy_interconnect = 0
        self.Total_energy_fetch = 0

        # Event queue
        self.data_transfer_erp = []

        # Event queue's PE index
        self.edram_pe_idx        = set()
        self.cu_operation_pe_idx = set()
        self.pe_saa_pe_idx       = set()
        self.activation_pe_idx   = set()
        self.edram_wr_pe_idx     = set()
        self.pooling_pe_idx      = set()

        self.Trigger = dict()

        # Utilization
        if self.record_PE:
            self.pe_state_for_plot = [0] # [{PE1, PE2}, {PE1, PE2}, {PE1}, {PE1}, ...]
        if self.record_layer:
            self.layer_state_for_plot = [0] # [{layer0, layer1}, {layer0}, {layer0, layer1}...]
        
        # Pipeline control
        if not self.isPipeLine: # Non_pipeline
            self.pipeline_layer_stage = 0
            self.events_each_layer = []
            self.edram_n = 0
            for layer in range(self.ordergenerator.model_info.layer_length):
                self.events_each_layer.append(0)
            for e in self.Computation_order:
                self.events_each_layer[e.nlayer] += 1

            self.this_layer_event_ctr = 0
            self.this_layer_cycle_ctr = 0
            self.cycles_each_layer = []
            print("events_each_layer:", self.events_each_layer)
            self.Non_pipeline_trigger = []

        self.t_edram, self.t_cuop, self.t_pesaa = 0, 0, 0
        self.t_act,      self.t_pool  = 0, 0
        self.t_transfer, self.t_fetch = 0, 0
        self.t_trigger,  self.t_inter = 0, 0
        self.cycle_ctr = 0
        self.done_event = 0

        self.transfer_cycles = []
        self.transfer_data_fm = []
        self.transfer_data_inter = []
        self.fetch_data = []
        for nlayer in range(self.ordergenerator.model_info.layer_length):
            self.transfer_cycles.append(0)
            self.transfer_data_fm.append(0)
            self.transfer_data_inter.append(0)
            self.fetch_data.append(0)
        self.busy_xb = 0

        self.event_fetch_ctr = dict()

        self.run()
        self.print_statistics_result()

    def run(self):
        print("estimation start")
        for event in self.Computation_order:
            if event.nlayer != 0:
                break
            if event.preceding_event_count == 0:
                # append first layer's edram_rd_ir event
                pos = event.position_idx
                pe_idx = pos[:-1]
                pe = self.PE_array[pe_idx]
                pe.edram_erp.append(event)
                if len(pe.edram_erp) > 1:
                    print("error: edram_erp len > 1")
                    exit()
                self.edram_pe_idx.add(pe)
        
        while True:
            if self.cycle_ctr % 10000 == 0:
                print("-----------------------------------------------------------------------")
                print("Completed: {} %".format(int(self.done_event/len(self.Computation_order) * 100)))
                print("Model: {}".format(self.model_config.Model_type))
                print("Mapping: {}, Scheduling: {}".format(self.mapping_str, self.scheduling_str))
                print("Cycle: {}, Done event: {}".format(self.cycle_ctr, self.done_event))
                print()
                print("edram: {:.6f}, cuop : {:.6f}, pesaa:    {:.6f}, act  : {:.6f}".format(self.t_edram, self.t_cuop, self.t_pesaa, self.t_act))
                print("pool : {:.6f}, transfer: {:.6f}, fetch: {:.6f}".format(self.t_pool, self.t_transfer, self.t_fetch))
                print("trigger: {:.6f}, interconnect: {:.6f}".format(self.t_trigger, self.t_inter))
                self.t_edram, self.t_cuop, self.t_pesaa = 0, 0, 0
                self.t_act,      self.t_pool  = 0, 0
                self.t_transfer, self.t_fetch = 0, 0
                self.t_trigger,  self.t_inter = 0, 0

            self.cycle_ctr += 1

            if self.record_PE:
                if len(self.pe_state_for_plot) == self.cycle_ctr:
                    self.pe_state_for_plot.append(set())
            if self.record_layer:
                if len(self.layer_state_for_plot) == self.cycle_ctr:
                    self.layer_state_for_plot.append(set())

            ## Pipeline stage control ###
            if not self.isPipeLine: # Non_pipeline
                self.this_layer_cycle_ctr += 1
                if self.this_layer_event_ctr == self.events_each_layer[self.pipeline_layer_stage] and not self.interconnect.busy_router:
                    self.pipeline_layer_stage += 1
                    print("\n------Layer", self.pipeline_layer_stage)
                    self.cycles_each_layer.append(self.this_layer_cycle_ctr)
                    self.this_layer_cycle_ctr = 0
                    self.this_layer_event_ctr = 0
                    
                    for trigger in self.Non_pipeline_trigger:
                        pe = trigger[0]
                        event = trigger[1]
                        pe.edram_erp.append(event)
                        if len(pe.edram_erp) > 1:
                            print("error: edram_erp len > 1")
                            exit()
                        self.edram_pe_idx.add(pe)
                    
                    self.Non_pipeline_trigger = []

            if self.trace:
                print("cycle:", self.cycle_ctr)
            self.trigger_event()
            self.event_edram()
            self.event_cu_op()
            self.event_pe_saa()
            self.event_act()
            self.event_pooling()
            self.event_transfer()
            self.interconnect_fn()
            self.fetch()

            # self.record_buffer_util()
            
            ### Finish
            if self.done_event == len(self.Computation_order):
                break

    def trigger_event(self):
        tt = time.time()
        if self.cycle_ctr in self.Trigger:
            for trigger in self.Trigger[self.cycle_ctr]:
                pe, event = trigger[0], trigger[1]
                if self.trace:
                    print(f"\t Trigger[{self.cycle_ctr}]: {trigger}")
                
                if event.event_type == "edram_rd_ir" or event.event_type == "edram_rd" or event.event_type == "edram_wr":
                    pe.edram_erp.append(event)
                    if len(pe.edram_erp) > 1:
                        print("error: edram_erp len > 1")
                        exit()
                    self.edram_pe_idx.add(pe)
                    
                elif event.event_type == "cu_operation":
                    # 這邊比較特別處理, 分成trigger cu operation和cu operation完成
                    if event.preceding_event_count == event.current_number_of_preceding_event:
                        # cu_op start
                        pe.cu_op_erp = event
                        self.cu_operation_pe_idx.add(pe)
                    elif event.preceding_event_count == (event.current_number_of_preceding_event - 1):
                        # cu_op end
                        for proceeding_index in event.proceeding_event:
                            pro_event = self.Computation_order[proceeding_index]
                            pro_event.current_number_of_preceding_event += 1
                            if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                                if pro_event.event_type == "pe_saa":
                                    pe.pe_saa_erp.append(pro_event)
                                    self.pe_saa_pe_idx.add(pe)
                                elif pro_event.event_type == "edram_rd_ir":
                                        pe.edram_erp.append(pro_event)
                                        if len(pe.edram_erp) > 1:
                                            print("error: edram_erp len > 1")
                                            exit()
                                        self.edram_pe_idx.add(pe)
                    
                elif event.event_type == "activation":
                    pe.activation_erp.append(event)
                    self.activation_pe_idx.add(pe)
                
                elif event.event_type == "data_transfer":
                    self.data_transfer_erp.append(event)
                                
                elif event.event_type == "pooling":
                    pe.pooling_erp.append(event)
                    self.pooling_pe_idx.add(pe)

                elif event.event_type == "pe_saa":
                    pe.pe_saa_erp.append(event)
                    self.pe_saa_pe_idx.add(pe)
                
                else:
                    print("error event type:", event.event_type)
                    exit()

            del self.Trigger[self.cycle_ctr]
        
        self.t_trigger += time.time() - tt

    def event_edram(self):
        tt = time.time()
        check_pe_idx = set()
        for pe in self.edram_pe_idx:
            event = pe.edram_erp.popleft()
            if event.event_type == "edram_wr":
                if self.trace:
                    print("\tdo edram_wr event_idx:", self.Computation_order.index(event))
                
                ### log
                pe.edram_event_order.append(self.Computation_order.index(event))

                self.done_event += 1
                if not self.isPipeLine:
                    self.this_layer_event_ctr += 1
                
                # Energy
                edram_write_data = event.outputs
                num_data = len(edram_write_data)
                pe.Bus_energy += self.hw_config.Energy_bus * self.input_bit * num_data
                pe.eDRAM_buffer_energy += self.hw_config.Energy_edram_buffer * self.input_bit * num_data

                # Write data
                for data in edram_write_data:
                    K = pe.edram_buffer.put(data, data)
                    if K: # kick data out of buffer
                        # TODO: simulate data transfer
                        pe.eDRAM_buffer_energy += self.hw_config.Energy_edram_buffer * self.input_bit # read
                        transfer_distance = pe.position[0]
                        self.Total_energy_interconnect += self.hw_config.Energy_router * self.input_bit * (transfer_distance + 1)
                        self.Total_energy_interconnect += self.hw_config.Energy_link   * self.input_bit * (transfer_distance + 1)
                        self.Total_energy_fetch += self.hw_config.Energy_off_chip_Wr * self.input_bit

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

                if self.record_PE: # PE util
                    self.pe_state_for_plot[self.cycle_ctr].add(pe)

                if self.record_layer: # layer
                    self.layer_state_for_plot[self.cycle_ctr].add(event.nlayer)
                
                if pe.edram_erp:
                    check_pe_idx.add(pe)

            elif event.event_type == "edram_rd_ir":
                edram_rd_data = event.inputs
                # check data
                Fetch_data = []
                for data in edram_rd_data:
                    if not pe.edram_buffer.get(data):
                        Fetch_data.append(data)
                        pe.edram_buffer.miss += 1
                if Fetch_data:
                    if self.trace:
                        print("\tfetch event_idx:", self.Computation_order.index(event))

                    event.current_number_of_preceding_event -= 1
                    finish_cycle = self.cycle_ctr + 1
                    if self.trace:
                        print("\t\tevent: ", event)
                        print("\t\tfinish_cycle: ", finish_cycle)
                    if finish_cycle in self.fetch_dict:
                        self.fetch_dict[finish_cycle].append([event, Fetch_data])
                    else:
                        self.fetch_dict[finish_cycle] = [[event, Fetch_data]]
                else:
                    if self.trace:
                        print("\tdo edram_rd_ir event_idx:", self.Computation_order.index(event),", layer", event.nlayer)
                    
                    ### log
                    pe.edram_event_order.append(self.Computation_order.index(event))
                    
                    self.done_event += 1
                    if not self.isPipeLine:
                        self.this_layer_event_ctr += 1

                    # Energy
                    num_data = len(edram_rd_data)
                    pe.eDRAM_buffer_energy += self.hw_config.Energy_edram_buffer * self.input_bit * num_data # eDRAM read
                    pe.Bus_energy          += self.hw_config.Energy_bus          * self.input_bit * num_data # bus
                    pe.CU_IR_energy        += self.hw_config.Energy_ir_in_cu     * self.input_bit * num_data # IR write
                    
                    # Trigger: CU operation and edram
                    finish_cycle = self.cycle_ctr + self.hw_config.eDRAM_buffer_read_to_IR_cycles
                    for proceeding_index in event.proceeding_event:
                        pro_event = self.Computation_order[proceeding_index]
                        pro_event.current_number_of_preceding_event += 1

                        if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                            if not self.isPipeLine and pro_event.nlayer != self.pipeline_layer_stage: # Non_pipeline
                                self.Non_pipeline_trigger.append([pe, pro_event])
                            else:
                                if finish_cycle not in self.Trigger:
                                    self.Trigger[finish_cycle] = [[pe, pro_event]]
                                else:
                                    self.Trigger[finish_cycle].append([pe, pro_event])
                    
                    # PE util
                    if self.record_PE:
                        self.pe_state_for_plot[self.cycle_ctr].add(pe)

                    # FIXME: a127a127
                    #self.log[self.Computation_order.index(event)] = [self.cycle_ctr, finish_cycle + 1]
                    # layer
                    if self.record_layer:
                        self.layer_state_for_plot[self.cycle_ctr].add(event.nlayer)
                    
                    if pe.edram_erp:
                        check_pe_idx.add(pe)

            elif event.event_type == "edram_rd":
                edram_rd_data = event.inputs
                # check data
                Fetch_data = []
                for data in edram_rd_data:
                    if not pe.edram_buffer.get(data):
                        Fetch_data.append(data)
                        pe.edram_buffer.miss += 1
                if Fetch_data:
                    if self.trace:
                        print("\tfetch event_idx:", self.Computation_order.index(event))

                    event.current_number_of_preceding_event -= 1
                    finish_cycle = self.cycle_ctr + 1
                    if finish_cycle in self.fetch_dict:
                        self.fetch_dict[finish_cycle].append([event, Fetch_data])
                    else: 
                        self.fetch_dict[finish_cycle] = [[event, Fetch_data]]
                else:
                    if self.trace:
                        print("\tdo edram_rd event_idx:", self.Computation_order.index(event),", layer", event.nlayer)
                    
                    ### log
                    pe.edram_event_order.append(self.Computation_order.index(event))

                    self.done_event += 1
                    if not self.isPipeLine:
                        self.this_layer_event_ctr += 1
                    
                    # Energy
                    num_data = len(edram_rd_data)
                    pe.eDRAM_buffer_energy += self.hw_config.Energy_edram_buffer * self.input_bit * num_data # eDRAM read
                    pe.Bus_energy          += self.hw_config.Energy_bus          * self.input_bit * num_data # bus
                    
                    # Trigger: pooling and edram
                    finish_cycle = self.cycle_ctr + 1
                    for proceeding_index in event.proceeding_event:
                        pro_event = self.Computation_order[proceeding_index]
                        pro_event.current_number_of_preceding_event += 1

                        if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                            if not self.isPipeLine and pro_event.nlayer != self.pipeline_layer_stage: # Non_pipeline
                                self.Non_pipeline_trigger.append([pe, pro_event])
                            else:
                                if finish_cycle not in self.Trigger:
                                    self.Trigger[finish_cycle] = [[pe, pro_event]]
                                else:
                                    self.Trigger[finish_cycle].append([pe, pro_event])     
                        
                    # PE util
                    if self.record_PE:
                        self.pe_state_for_plot[self.cycle_ctr].add(pe)

                    # layer
                    if self.record_layer:
                        self.layer_state_for_plot[self.cycle_ctr].add(event.nlayer)
            
                    if pe.edram_erp:
                        check_pe_idx.add(pe)

        self.edram_pe_idx = check_pe_idx
        self.t_edram += time.time() - tt

    def event_cu_op(self):
        tt = time.time()
        for pe in self.cu_operation_pe_idx:
            event = pe.cu_op_erp
            pe.cu_op_erp = 0
            
            if self.trace:
                print("\tcu_operation event_idx:", self.Computation_order.index(event))
            
            self.done_event += 1
            if not self.isPipeLine:
                self.this_layer_event_ctr += 1
            
            # Energy
            ou_num_dict = event.inputs[1]
            for xb_idx in ou_num_dict:
                ou_num = ou_num_dict[xb_idx]
                pe.CU_dac_energy           += self.hw_config.Energy_ou_dac      * ou_num
                pe.CU_crossbar_energy      += self.hw_config.Energy_ou_crossbar * ou_num
                pe.CU_adc_energy           += self.hw_config.Energy_ou_adc      * ou_num
                pe.CU_shift_and_add_energy += self.hw_config.Energy_ou_ssa      * ou_num
                
                pe.CU_IR_energy            += self.hw_config.Energy_ir_in_cu    * ou_num * self.hw_config.OU_h
                pe.CU_OR_energy += self.hw_config.Energy_or_in_cu * ou_num * self.hw_config.OU_w * self.hw_config.ADC_resolution

                self.busy_xb += ou_num
                
            event.current_number_of_preceding_event += 1 # if current_number_of_preceding_event == 2: cu_op finish
            total_cycles = event.inputs[0]
            finish_cycle = self.cycle_ctr + total_cycles + 2 # +2: pipeline last two stage

            # FIXME: a127a127
            #self.log[self.Computation_order.index(event)] = [self.cycle_ctr, finish_cycle + 1]

            if hasattr(event, 'window_id'):
                if not 'window_event' in self.log:
                    self.log['window_event'] = []
                self.log['window_event'].append((self.cycle_ctr, finish_cycle, event))
            if finish_cycle not in self.Trigger:
                self.Trigger[finish_cycle] = [[pe, event]]
            else:
                self.Trigger[finish_cycle].append([pe, event])
                        
            if self.record_PE: # PE util
                for cycle in range(len(self.pe_state_for_plot), finish_cycle+1):
                    self.pe_state_for_plot.append(set())
                for cycle in range(self.cycle_ctr, finish_cycle+1):
                    self.pe_state_for_plot[cycle].add(pe)
            
            if self.record_layer: # layer
                for cycle in range(len(self.layer_state_for_plot), finish_cycle):
                    self.layer_state_for_plot.append(set())
                for cycle in range(self.cycle_ctr, finish_cycle):
                    self.layer_state_for_plot[cycle].add(event.nlayer)
            
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

            # FIXME: a127a127
            #self.log[self.Computation_order.index(event)] = [self.cycle_ctr, self.cycle_ctr + 1]
            
            # Energy
            saa_amount = event.inputs
            pe.Or_energy += self.hw_config.Energy_or * self.input_bit * saa_amount
            pe.Bus_energy += self.hw_config.Energy_bus * self.input_bit * saa_amount
            pe.PE_shift_and_add_energy += self.hw_config.Energy_shift_and_add_in_PE * saa_amount
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
            
            if self.record_PE: # PE util
                self.pe_state_for_plot[self.cycle_ctr].add(pe)

            if self.record_layer: # layer
                self.layer_state_for_plot[self.cycle_ctr].add(event.nlayer)

            if pe.pe_saa_erp:
                check_pe_idx.add(pe)
            
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

            # FIXME: a127a127
            #self.log[self.Computation_order.index(event)] = [self.cycle_ctr, self.cycle_ctr + 1]
            
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
            
            if self.record_PE: # PE util
                self.pe_state_for_plot[self.cycle_ctr].add(pe)

            if self.record_layer: # layer
                self.layer_state_for_plot[self.cycle_ctr].add(event.nlayer)

            if pe.activation_erp:
                check_pe_idx.add(pe)

        self.activation_pe_idx = check_pe_idx

        self.t_act += time.time() - tt

    def event_pooling(self):
        tt = time.time()
        check_pe_idx = set()
        for pe in self.pooling_pe_idx:
            event = pe.pooling_erp.popleft()
            if self.trace:
                print("\tdo pooling event_idx:", self.Computation_order.index(event))

            # FIXME: a127a127
            #self.log[self.Computation_order.index(event)] = [self.cycle_ctr, self.cycle_ctr + 1]
            if not 'pooling' in self.log:
                self.log['pooling'] = []
            self.log['pooling'].append((self.cycle_ctr, event))
            
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
            
            if self.record_PE: # PE util
                self.pe_state_for_plot[self.cycle_ctr].add(pe)

            if self.record_layer: # layer
                self.layer_state_for_plot[self.cycle_ctr].add(event.nlayer)

            if pe.pooling_erp:
                check_pe_idx.add(pe)
            
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
            des_pe = self.PE_array[data_transfer_des]

            # FIXME: a127a127
            if not 'data_transfer' in self.log:
                self.log['data_transfer'] = []

            if data_transfer_src == data_transfer_des:
                # Trigger
                finish_cycle = self.cycle_ctr + 1

                # FIXME: a127a127
                #self.log[self.Computation_order.index(event)] = [self.cycle_ctr, self.cycle_ctr + 1]
                self.log['data_transfer'].append((self.cycle_ctr, transfer_data, data_transfer_src[0:2], data_transfer_des[0:2]))

                for data in transfer_data:
                    K = des_pe.edram_buffer.put(data, data)
                    if K: # kick data out of buffer
                        # TODO: simulate data transfer
                        des_pe.eDRAM_buffer_energy += self.hw_config.Energy_edram_buffer * self.input_bit # read
                        transfer_distance = data_transfer_des[0]
                        self.Total_energy_interconnect += self.hw_config.Energy_router * self.input_bit * (transfer_distance + 1)
                        self.Total_energy_interconnect += self.hw_config.Energy_link   * self.input_bit * (transfer_distance + 1)
                        self.Total_energy_fetch += self.hw_config.Energy_off_chip_Wr * self.input_bit

                for pro_event_idx in event.proceeding_event:
                    pro_event = self.Computation_order[pro_event_idx]
                    pro_event.current_number_of_preceding_event += 1
                    if pro_event.preceding_event_count == pro_event.current_number_of_preceding_event:
                        if not self.isPipeLine and pro_event.nlayer != self.pipeline_layer_stage: # Non_pipeline
                            self.Non_pipeline_trigger.append([des_pe, pro_event])
                        else:
                            if finish_cycle not in self.Trigger:
                                self.Trigger[finish_cycle] = [[des_pe, pro_event]]
                            else:
                                self.Trigger[finish_cycle].append([des_pe, pro_event])
            else:
                num_data = len(transfer_data)
                transfer_distance  = abs(data_transfer_des[1] - data_transfer_src[1])
                transfer_distance += abs(data_transfer_des[0] - data_transfer_src[0])

                # Energy
                self.Total_energy_interconnect += self.hw_config.Energy_router * self.input_bit * num_data * (transfer_distance + 1)
                self.Total_energy_interconnect += self.hw_config.Energy_link   * self.input_bit * num_data * (transfer_distance + 1)
                
                nlayer = transfer_data[0][0]
                if len(transfer_data[0]) == 4:
                    self.transfer_data_fm[nlayer] += num_data
                else:
                    self.transfer_data_inter[nlayer] += num_data

                for i in range(len(transfer_data)-1):
                    data = transfer_data[i]
                    packet = Packet(data_transfer_src, data_transfer_des, data, [], self.cycle_ctr)
                    self.interconnect.input_packet(packet)
                data = transfer_data[-1]
                packet = Packet(data_transfer_src, data_transfer_des, data, event.proceeding_event, self.cycle_ctr)
                self.interconnect.input_packet(packet)

                # FIXME: a127a127
                #self.log[self.Computation_order.index(event)] = [self.cycle_ctr, self.cycle_ctr + transfer_distance]
        self.data_transfer_erp = []

        self.t_transfer += time.time() - tt

    def fetch(self):
        tt = time.time()
        if self.cycle_ctr in self.fetch_dict:
            fetch_list = self.fetch_dict[self.cycle_ctr]
            if self.trace:
                print("\t in fetch_dict")
                print("\t fetch_list: ", fetch_list)
            for F in fetch_list:
                event, transfer_data = F[0], F[1]
                event_idx = self.Computation_order.index(event)
                num_data = len(transfer_data)
                rty, rtx = event.position_idx[0], event.position_idx[1]
                pey, pex = event.position_idx[2], event.position_idx[3]
                data_transfer_src = (0, rtx)
                data_transfer_des = (rty, rtx, pey, pex)

                # Energy
                self.Total_energy_fetch += self.hw_config.Energy_off_chip_Rd * self.input_bit * num_data

                for i in range(len(transfer_data)-1):
                    data = transfer_data[i]
                    packet = Packet(data_transfer_src, data_transfer_des, data, [], self.cycle_ctr)
                    self.interconnect.input_packet(packet)
                data = transfer_data[-1]
                packet = Packet(data_transfer_src, data_transfer_des, data, [event_idx], self.cycle_ctr)
                self.interconnect.input_packet(packet)

                # Statistic
                nlayer = event.nlayer
                if len(transfer_data[0]) == 4:
                    self.transfer_data_fm[nlayer] += num_data
                else:
                    self.transfer_data_inter[nlayer] += num_data
                self.fetch_data[nlayer] += num_data

        self.t_fetch += time.time() - tt

    def interconnect_fn(self):
        tt = time.time()
        if self.interconnect.busy_router:
            for s in range(self.hw_config.interconnect_step_num):
                arrived, data_transfers = self.interconnect.step()
                for data_transfer in data_transfers:
                    self.log['data_transfer'].append((self.cycle_ctr, [data_transfer[2].data], data_transfer[0], data_transfer[1]))
                for packet in arrived:
                    des_pe_id = packet.destination
                    des_pe = self.PE_array[des_pe_id]
                    data = packet.data
                    pro_event_list = packet.pro_event_list

                    K = des_pe.edram_buffer.put(data, data)
                    if K: # kick data out of buffer
                        # TODO: simulate data transfer
                        des_pe.eDRAM_buffer_energy += self.hw_config.Energy_edram_buffer * self.input_bit # read
                        transfer_distance = des_pe_id[0]
                        self.Total_energy_interconnect += self.hw_config.Energy_router * self.input_bit * (transfer_distance + 1)
                        self.Total_energy_interconnect += self.hw_config.Energy_link   * self.input_bit * (transfer_distance + 1)
                        self.Total_energy_fetch += self.hw_config.Energy_off_chip_Wr * self.input_bit

                    # Energy
                    des_pe.eDRAM_buffer_energy += self.hw_config.Energy_edram_buffer * self.input_bit # write
                    start_transfer_cycle = packet.start_transfer_cycle
                    end_transfer_cycle = self.cycle_ctr + 1
                    nlayer = data[0]
                    self.transfer_cycles[nlayer] += end_transfer_cycle - start_transfer_cycle

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

        self.t_inter += time.time() - tt

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
        self.eDRAM_buffer_energy     = 0.
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
            self.eDRAM_buffer_energy     += pe.eDRAM_buffer_energy
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
        self.Total_energy = self.eDRAM_buffer_energy + self.Bus_energy + self.PE_shift_and_add_energy + \
                            self.Or_energy + self.Activation_energy + self.Pooling_energy + \
                            self.CU_shift_and_add_energy + self.CU_dac_energy + self.CU_adc_energy + \
                            self.CU_crossbar_energy + self.CU_IR_energy + self.CU_OR_energy + \
                            self.Total_energy_interconnect + self.Total_energy_fetch

        self.output_result()
        self.buffer_analysis()
        # self.PE_energy_breakdown()
        self.miss_rate()
        if self.record_PE: # PE util
            print("output pe utilization...")
            self.pe_utilization()
        if self.record_layer: # layer
            print("output layer utilization...")
            self.layer_utilization()
        
    def output_result(self):
        overlap_layer_ctr = 0
        layers_per_cycle_ctr = 0
        if self.record_layer: # layer
            for cycle in range(1, len(self.layer_state_for_plot)):
                if len(self.layer_state_for_plot[cycle]) > 1:
                    overlap_layer_ctr += 1
                layers_per_cycle_ctr += len(self.layer_state_for_plot[cycle])

        
        with open(self.path+'/Result.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["overlap_layer_ctr", overlap_layer_ctr])
            writer.writerow(["overlap_layer_ctr/Cycles", overlap_layer_ctr/self.cycle_ctr])
            writer.writerow(["layers_per_cycle_ctr", layers_per_cycle_ctr/self.cycle_ctr])
        
            writer.writerow([])
            writer.writerow(["Cycles", self.cycle_ctr])
            writer.writerow(["Energy(nJ)", self.Total_energy])

            writer.writerow([])
            writer.writerow(["", "Energy(nJ)"])
            writer.writerow(["Interconnect", self.Total_energy_interconnect])
            writer.writerow(["Fetch", self.Total_energy_fetch])

            writer.writerow(["eDRAM Buffer", self.eDRAM_buffer_energy])
            writer.writerow(["Bus", self.Bus_energy])
            writer.writerow(["PE Shift and Add", self.PE_shift_and_add_energy])
            writer.writerow(["PE's OR", self.Or_energy])
            writer.writerow(["Activation", self.Activation_energy])
            writer.writerow(["Pooling", self.Pooling_energy])

            writer.writerow(["CU Shift and Add", self.CU_shift_and_add_energy])
            writer.writerow(["DAC", self.CU_dac_energy])
            writer.writerow(["ADC", self.CU_adc_energy])
            writer.writerow(["Crossbar Array", self.CU_crossbar_energy])
            writer.writerow(["IR", self.CU_IR_energy])
            writer.writerow(["CU's OR", self.CU_OR_energy])

            writer.writerow([])
            writer.writerow(["Busy XB", self.busy_xb])
            writer.writerow(["Avg", self.busy_xb/self.cycle_ctr])

            writer.writerow([])
            fm_num  = self.ordergenerator.transfer_feature_map_data_num
            inter_num = self.ordergenerator.transfer_intermediate_data_num
            writer.writerow(["transfer data between PE"])
            writer.writerow(["feature map data", fm_num])
            writer.writerow(["intermediate data", inter_num])

            writer.writerow([])
            writer.writerow(["Communication:"])
            writer.writerow(["layer", "transfer cycles", "transfer data fm", "transfer data inter", "Average", "", "fetch data"])
            tc, td_f, td_i, fd, tavg = 0, 0, 0, 0, 0
            for nlayer in range(self.ordergenerator.model_info.layer_length):
                tc += self.transfer_cycles[nlayer]
                td_f += self.transfer_data_fm[nlayer]
                td_i += self.transfer_data_inter[nlayer]
                fd += self.fetch_data[nlayer]
                if self.transfer_data_fm[nlayer] != 0 and self.transfer_data_inter[nlayer] != 0:
                    avg = self.transfer_cycles[nlayer]/(self.transfer_data_fm[nlayer]+self.transfer_data_inter[nlayer])
                    tavg += avg
                else:
                    avg = 0
                writer.writerow([nlayer, self.transfer_cycles[nlayer], self.transfer_data_fm[nlayer], self.transfer_data_inter[nlayer], avg, "", self.fetch_data[nlayer]])
            writer.writerow(["total", tc, td_f, td_i, tc/(td_f+td_i), "", fd])
            writer.writerow([])
            writer.writerow(["", "Event"])
            writer.writerow(["Total", len(self.Computation_order)])
            writer.writerow(["edram_rd_ir", self.ordergenerator.edram_rd_ir_ctr])
            writer.writerow(["cu_op", self.ordergenerator.cu_op_ctr])
            writer.writerow(["pe_saa", self.ordergenerator.pe_saa_ctr])
            writer.writerow(["edram_wr", self.ordergenerator.edram_wr_ctr])
            writer.writerow(["edram_rd", self.ordergenerator.edram_rd_ctr])
            writer.writerow(["pooling", self.ordergenerator.pooling_ctr])
            writer.writerow(["data_transfer", self.ordergenerator.data_transfer_ctr])

        ### log
        with open(self.path+'/event_order_log.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["PE"])
            for pe_pos in self.PE_array:
                pe = self.PE_array[pe_pos]
                A = ""
                if pe.edram_event_order:
                    idx1 = -1
                    for idx in pe.edram_event_order:
                        if idx <= idx1:
                            A = "Order error"
                            break
                        idx1 = idx
                    writer.writerow([pe.plot_idx, A, str(pe.edram_event_order)])

    def PE_energy_breakdown(self):
        # PE breakdown
        print("output PE energy breakdown")
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
        with open(self.path+'/Buffer_utilization.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["", "SUM(KB)", "", "tmp_data", "fm_data"])
            for pe_pos in self.PE_array:
                pe  = self.PE_array[pe_pos]
                buf = pe.edram_buffer.buffer
                tmp_data, fm_data = 0, 0
                for data in buf:
                    if len(data) == 5: # intermediate data
                        tmp_data += 1
                    elif len(data) == 4: # feature map data
                        fm_data += 1
                tmp_data *= self.input_bit/8/1024
                fm_data *= self.input_bit/8/1024
                summ = tmp_data + fm_data
                if summ == 0:
                    continue
                rty, rtx, pey, pex = pe_pos[0], pe_pos[1], pe_pos[2], pe_pos[3]
                idx = pex + pey * self.hw_config.PE_num_x + rtx * self.hw_config.PE_num + rty * self.hw_config.PE_num * self.hw_config.Router_num_x
                writer.writerow(["PE"+str(idx), summ, "", tmp_data, fm_data]) 

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

    def miss_rate(self):
        with open(self.path+'/Miss_rate.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for pe_pos in self.PE_array:
                pe = self.PE_array[pe_pos]
                if len(pe.edram_buffer.buffer) != 0:
                    rty, rtx, pey, pex = pe_pos[0], pe_pos[1], pe_pos[2], pe_pos[3]
                    idx = pex + pey * self.hw_config.PE_num_x + rtx * self.hw_config.PE_num + rty * self.hw_config.PE_num * self.hw_config.Router_num_x
                    arr = ["PE"+str(idx)]

                    miss = pe.edram_buffer.miss
                    
                    arr.append(miss)
                    writer.writerow(arr)

    def pe_utilization(self):
        with open(self.path+'/PE_utilization.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for cycle in range(1, len(self.pe_state_for_plot)):
                if self.pe_state_for_plot[cycle]:
                    for pe in self.pe_state_for_plot[cycle]:
                        plot_idx = pe.plot_idx
                        writer.writerow([cycle, plot_idx])

    def layer_utilization(self):
        with open(self.path+'/Layer_utilization.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for cycle in range(1, len(self.layer_state_for_plot)):
                if self.layer_state_for_plot[cycle]:
                    for layer in self.layer_state_for_plot[cycle]:
                        writer.writerow([cycle, layer])
    
