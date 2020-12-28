from collections import OrderedDict

class OnChipBuffer(): # LRU
    def __init__(self, size):
        self.size = size
        self.buffer = OrderedDict()
        self.miss = 0
        
    def put(self, key, value):
        if key in self.buffer:
            self.buffer.pop(key)
            self.buffer.update({key: value})
            return None
        else:
            if self.size == len(self.buffer): # full
                self.buffer.popitem(last=False)
                self.buffer.update({key: value})
                return True
            else:
                self.buffer.update({key: value})
                return None

    def get(self, key):
        value = self.buffer.get(key)
        if not value:
            return None
        self.buffer.pop(key)
        self.buffer.update({key: value})
        return value
    
    def __str__(self):
        return str(self.buffer)