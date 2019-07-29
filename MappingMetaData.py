class MappingMetaData(object):
    def __init__(self, eventtype, nlayer, xbar_row, xbar_column, inputs):
        self.eventtype = eventtype
        self.nlayer = nlayer
        self.xbar_row = xbar_row
        self.xbar_column = xbar_column
        self.inputs = inputs

    def __str__(self):
        return str(self.__dict__)
        