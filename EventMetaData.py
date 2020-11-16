class EventMetaData:
    #__slots__ = ["event_type", "preceding_event_count", "current_number_of_preceding_event",
    #             "proceeding_event", "nlayer", "inputs", "outputs", 
    #             "position_idx"
    #            ]
    def __init__(self, event_type, position_idx, preceding_event_count, proceeding_event, nlayer, inputs, outputs):
        self.event_type = event_type
        self.preceding_event_count = preceding_event_count
        self.current_number_of_preceding_event = 0
        self.proceeding_event = proceeding_event
        self.nlayer = nlayer
        self.inputs = inputs
        self.outputs = outputs
        self.position_idx = position_idx

        # self.fetch = False
        # self.data_is_transfer = 0

    def __str__(self):
        return str({"event_type:": self.event_type, "preceding_event_count:": self.preceding_event_count, "current_number_of_preceding_event": self.current_number_of_preceding_event,
                    "proceeding_event:": self.proceeding_event, "nlayer:": self.nlayer, "inputs:": self.inputs, "outputs:": self.outputs,
                    "position_idx:": self.position_idx
                    })
    
    # event_type: edram_rd_ir
        # inputs : edram_read_data
        # outputs: 0
    
    # event_type: cu_operation
        # inputs : max_ou 
        # outputs: num_ou_in_xb # {XB1: 4 OUs, XB2: 3 OUs}
    
    # event_type: pe_saa
        # inputs : saa_amount
        # outputs: 0
    
    # event_type: activation
        # inputs : act_amount
        # outputs: 0
    
    # event_type: edram_wr
        # inputs : 0
        # outputs: edram_write_data
    
    # event_type: data_transfer
        # inputs : 0
        # outputs: transfer_data
        # position_idx: [data_transfer_src, data_transfer_des]
    
    # event_type: edram_rd
        # inputs :  edram_read_data
        # outputs:  0
    
    # event_type: pooling
        # inputs :  pooling_amount
        # outputs:  0
        
