class XB(object):
    def __init__(self, xb_pos):
        ## events per cycle
        # TODO: 隨給定資料調整
        self.ou_epc = 1

        ## trigger event list
        self.adc_trigger = []
        
        ## event ready pool
        self.ou_erp = []
        
        self.reset()
        
        ### for order generator
        self.position = xb_pos
        self.crossbar_array = []
        self.Convolution = []
        self.Fully = []

    def reset(self):
        ## state
        self.state_ou = [False] * self.ou_epc

    def check_state(self):
        if True in self.state_ou:
            return True
        return False

    def __str__(self):
        return str(self.__dict__)