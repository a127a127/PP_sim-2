class CrossbarGridMetaData(object):
    def __init__(self, nlayer, ngrid, nfilter, nbit):
        self.nlayer = nlayer
        self.nfilter = nfilter
        self.ngrid = ngrid
        self.nbit = nbit

    def __str__(self):
        return str(self.__dict__)