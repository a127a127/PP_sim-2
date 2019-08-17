class XB(object):
    def __init__(self, xb_pos):
        ### events per cycle
        # TODO: 隨給定資料調整
        self.ou_operation_epc = 1

        ### for mapping
        self.position = xb_pos

        ### trigger event list
        self.cu_saa_trigger = [] # [pro_event, [cu_idx]]
        
        ### event ready pool
        self.ou_operation_erp = []
        
        self.reset()
        
        ### for mapping
        self.position = xb_pos
        self.crossbar_array = []
        self.Convolution = []
        self.Fully = []
        self.Computation_order = []

    def reset(self):
        ### state
        self.state_ou_operation = [False] * self.ou_operation_epc

    def check_state(self):
        if True in self.state_ou_operation:
            return True
        return False

    def __str__(self):
        return str(self.__dict__)