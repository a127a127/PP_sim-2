class XB(object):
    def __init__(self, xb_pos):
        ### events per cycle
        # ### Todo: 隨給定資料調整
        self.ou_operation_epc = 1

        ### for mapping
        self.position = xb_pos

        ### event ready pool
        self.ou_operation_erp = []
        
        self.reset()
    
    def reset(self):
        ### state
        self.state_ou_operation = [False] * self.ou_operation_epc
        ### trigger event list
        self.cu_saa_trigger = [] # [pro_event, [cu_idx]]


    def __str__(self):
        return str(self.__dict__)