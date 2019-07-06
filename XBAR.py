class XBAR(object):
    def __init__(self, xbar_w, xbar_h, xb_pos):
        ### for mapping
        self.position = xb_pos
        self.crossbar_array = [[0] * xbar_w] * xbar_h
        self.Convolution = []
        self.Fully = []
        self.Computation_order = []

        ### event ready pool
        self.xb_erp = []

        
    def __str__(self):
        return str(self.__dict__)