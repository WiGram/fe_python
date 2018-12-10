class gjrArch(object):
    def __init__(self, initPar = None, data = None, est = None, se = None, rSe = None, tStat = None, rtStat = None, mlVal = None):
        # Attributes
        self._data = data
        self._ar_order = ar_order
    
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if len(value) / 2.0 < self.ar_order:
            raise ValueError('Data does not have enough elements ' \
                                'to estimate parameters')
        
        self._data = value
    
    @property
    def ar_order(self):
        return self._ar_order

    def estimate(self):
        raise NotImplementedError('This has not yet been implemented')


ar = AR()
ar.__dict__ # Shows all attributes (and more, usually)