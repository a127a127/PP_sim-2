#from HardwareMetaData import HardwareMetaData

class XBAR(object):
    def __init__(self, xb_pos):
        ### for mapping
        self.position = xb_pos

        # xbar_h = HardwareMetaData().Xbar_h
        # xbar_w = HardwareMetaData().Xbar_w
        # self.crossbar_array = xbar_h * [[0] * xbar_w]
        self.crossbar_array = []
        self.Convolution = []
        self.Fully = []
        self.Computation_order = []
        
        #self.input_require = []

    def __str__(self):
        return str(self.__dict__)