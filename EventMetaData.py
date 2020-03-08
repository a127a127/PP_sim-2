class EventMetaData:
    __slots__ = ["event_type", "preceding_event_count", "current_number_of_preceding_event",
                 "proceeding_event", "nlayer", "inputs", "outputs", 
                 "position_idx", "fetch", "data_is_transfer"
                ]
    def __init__(self, event_type, position_idx, preceding_event_count, proceeding_event, nlayer, inputs, outputs):
        self.event_type = event_type
        self.preceding_event_count = preceding_event_count
        self.current_number_of_preceding_event = 0
        self.proceeding_event = proceeding_event
        self.nlayer = nlayer
        self.inputs = inputs
        self.outputs = outputs
        self.position_idx = position_idx

        self.fetch = False
        self.data_is_transfer = 0

    def __str__(self):
        return str({"event_type:": self.event_type, "preceding_event_count:": self.preceding_event_count, "current_number_of_preceding_event": self.current_number_of_preceding_event,
                    "proceeding_event:": self.proceeding_event, "nlayer:": self.nlayer, "inputs:": self.inputs, "outputs:": self.outputs,
                    "position_idx:": self.position_idx, "fetch:": self.fetch, "data_is_transfer:": self.data_is_transfer
                    })
        