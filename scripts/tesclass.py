class A():
    def __init__(self):
        self.a = 1
    
    def update(self):
        self.a +=1
    def show(self):
        print("A , A - ",self.a)

class B():
    def __init__(self):
        self.b = A()
        self.c = C(self.b)
    
    def updateB(self):
        self.b.a +=1
    
    def show(self):
        print("B , A - ",self.b.a)
        print("C , A - ",self.c.m.a)
    
class C():
    def __init__(self,X):
        self.m = X

    def update(self):
        self.m.a +=1
    
    def show(self):
        print("C1 , A - ",self.m.a)

A1 = A()

A1.show()

C1 = C(A1)
C1.show()
C1.update()
A1.show()
C1.show()
A1.update()
A1.show()
C1.show()
