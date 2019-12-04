class EventMetaData:
    __slots__ = ["event_type", "preceding_event_count", "current_number_of_preceding_event",
                 "proceeding_event", "nlayer", "inputs", "outputs", 
                 "position_idx", "finished", "data_is_transfer"
                ]
    def __init__(self, event_type, position_idx, preceding_event_count, proceeding_event, nlayer, inputs, outputs):
        
        self.event_type = event_type  # edram_rd_ir, ou, cu_saa, pe_saa, activation, edram_wr, edram_rd_pool, pooling, data_transfer
        
        self.preceding_event_count = preceding_event_count
        self.current_number_of_preceding_event = 0
        self.proceeding_event = proceeding_event

        self.nlayer = nlayer
        self.inputs = inputs
        self.outputs = outputs
        self.position_idx = position_idx

        self.finished = False

        self.data_is_transfer = 0

        # idle analysis
        # self.last_arrived_data = 0 # for edram read event
        # self.pre_edram_rd_idx = -1 # for ou event
        # self.xb_arrived_data = {}  # dict[xb_id]: [first_arrived, last_arrived]

    def __str__(self):
        return str(self.__dict__)
        