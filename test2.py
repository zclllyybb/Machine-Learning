class T:
    def __init__(self):
        self.w = 1

    def getW(self):
        print("self.W = %d" % self.w)
        return self.w


a = T()
ww = a.getW()
ww = 100
print(ww, a.getW())