class XB(object):
    def __init__(self, xb_pos):
        self.state_ou = False

        ## trigger event list
        self.adc_trigger = []
        
        ## event ready pool
        self.ou_erp = []
        
        ### for order generator
        self.position = xb_pos
        self.crossbar_array = []
        self.Convolution = []
        self.Fully = []

        ## for idle analysis
        # self.compute = 0
        # self.transfer = 0
        # self.wait_resource = 0
        # self.last_cycle_state = False
        # self.idle_to_busy = [1]
        # self.busy_to_idle = [1]

    def reset(self, cycle):
        ## state
        if self.state_ou:
            self.last_cycle_state = True
        else:
            if self.last_cycle_state:
                self.busy_to_idle.append(cycle)
            self.last_cycle_state = False
        self.state_ou = False

    def check_state(self):
        if self.state_ou:
            return True
        return False

    def __str__(self):
        return str(self.__dict__)