class FeatureMapMetaData(object):
    def __init__(self, start_compute, start_compute_pos, finish_compute):
        self.start_compute = start_compute
        self.start_compute_pos = start_compute_pos
        self.finish_compute = finish_compute
        self.arrived_buffer = dict()
        #self.finish_pe_idx

    def __str__(self):
        return str(self.__dict__)
