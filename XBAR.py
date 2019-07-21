class XBAR(object):
    def __init__(self, xb_pos):
        ### for mapping
        self.position = xb_pos
        self.crossbar_array = []
        self.Convolution = []
        self.Fully = []
        self.Computation_order = []
        
    def __str__(self):
        return str(self.__dict__)