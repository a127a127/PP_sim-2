class MappingMetaData(object):
    def __init__(self, inputs, Cols, Filters):
        self.inputs  = inputs
        self.Cols    = Cols
        self.Filters = Filters

    def __str__(self):
        return str(self.__dict__)
        