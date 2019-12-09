import collections
class XB(object):
    def __init__(self, xb_pos):
        self.state = False

        ## trigger event list
        self.adc_trigger = []
        
        ## event ready pool
        #self.ou_erp = []
        self.ou_erp = collections.deque()
        
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

    def reset(self):
        ## state
        # if self.state_ou:
        #     self.last_cycle_state = True
        # else:
        #     if self.last_cycle_state:
        #         self.busy_to_idle.append(cycle)
        #     self.last_cycle_state = False
        self.state = False

    def check_state(self):
        return self.state

    def __str__(self):
        return str(self.__dict__)