



class MIMOCollator:
    def __init__(self, mimo):
        self.mimo = mimo
    
    def __call__(self, batch):
        print(batch)
