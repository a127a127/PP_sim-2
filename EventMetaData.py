class EventMetaData:
    def __init__(self, event_type, position_idx, preceding_event_count, proceeding_event, nlayer, inputs, outputs):
        self.event_type = event_type  # edram_rd_ir, ou_operation, cu_saa, pe_saa, activation, edram_wr
        
        self.preceding_event_count = preceding_event_count
        self.current_number_of_preceding_event = 0
        self.proceeding_event = proceeding_event

        self.nlayer = nlayer
        self.inputs = inputs # pe_saa: [[window_h, window_w, nfilter, pre_CU_idx]] edram_rd: [[]]
        self.outputs = outputs #[(num_input, hinput, winput, cinput, input_bit), (filter_nfilter, filter_ngrid, filter_nbit)]
        self.position_idx = position_idx

        self.finished = False

    def __str__(self):
        return str(self.__dict__)