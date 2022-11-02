# ========================== 01 ==========================
class Vwap():
    time_window = 1 # one step
    def __init__(self, historical_data, running_data):
        self.historical_data = historical_data
        self.running_data = running_data
        
    @property
    def difference(self):
        pass
    
class StaticVwap():
    def __init__(self, historical_data, running_data):
        pass
    
class DynamicVwap():
    def __init__(self):
        pass

class VwapEstimator():
    def __init__(self):
        pass
    
# ========================== 02 ==========================
class ImplementedShortfall():
    def __init__(self):
        pass
    
    
class RelativePerformance_IS():
    def __init__(self):
        pass
    
    
# ========================== 03 ==========================
class AlmgrenChriss():
    def __init__(self):
        pass
    
    def u(self,t):
        return 0#TODO
    def q(self,t):
        return 0#TODO
    def S(self,t):
        return 0#TODO
    def S_tilde(self,t):
        return 0#TODO
    
    @property
    def book_value_at_initial_time(self):
        return q(0) * S(0)
    @property
    def running_revenue(self):
        '''self.time'''
        if t != T: return u(t) * S_tilde(t)
        else: return q(T) * S_tilde(T)
    @property  
    def revenue(self):
        revenue = 0
        for t in time_list:
            revenue += self.running_revenue
        return revenue
    @property  
    def total_execution_cost(self):
        return self.book_value_at_initial_time - self.revenue