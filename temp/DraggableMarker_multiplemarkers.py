import numpy as np
#import matplotlib.pyplot as plt
#
#x = np.arange(60)
#y = np.sin(x)*np.log(x+1)
#
#fig, ax = plt.subplots()
#ax.plot(x,y, marker="o", ms=4)

class DraggableMarker():
    def __init__(self,ax, lines=None):
        self.ax = ax
        if lines==None:
            self.lines=self.ax.lines
        else:
            self.lines=lines
        self.lines = self.lines[:]
        self.tx =  [self.ax.text(l.get_xdata()[0],l.get_ydata()[0],"") for l in self.lines]
        self.marker = [self.ax.plot([l.get_xdata()[0]],[l.get_ydata()[0]], marker="o", color="red")[0]  for l in self.lines]

        self.draggable=False
        
        self.c1 = self.ax.figure.canvas.mpl_connect("button_press_event", self.click)
        self.c2 = self.ax.figure.canvas.mpl_connect("button_release_event", self.release)
        self.c3 = self.ax.figure.canvas.mpl_connect("motion_notify_event", self.drag)
        
        self.static = False
    
    def click(self,event):
        if event.button==1:
            #leftclick
            self.draggable=True
            self.update(event)
        elif event.button==3:
            self.draggable=False
        [tx.set_visible(self.draggable) for tx in self.tx]
        [m.set_visible(self.draggable) for m in self.marker]
        self.ax.figure.canvas.draw_idle()        
                
    def drag(self, event):
        if self.draggable:
            self.update(event)
            self.ax.figure.canvas.draw_idle()

    def release(self,event):
        self.draggable=False
        
    def update(self, event):
        if not self.static:
            for i, line in enumerate(self.lines):
                x,y = self.get_closest(line, event.xdata) 
                self.tx[i].set_position((x,y))
                self.tx[i].set_text("x:{}\ny:{:.3f}".format(x,y))
                self.marker[i].set_data([x],[y])
            
    def get_closest(self,line, mx):
        x,y = line.get_data()
        mini = np.argmin(np.abs(x-mx))
        return x[mini], y[mini]
    
    def set_static(self, st):
        self.static = st
    
    def findmax(self):
        for i, line in enumerate(self.lines):
            xd,yd = line.get_data()
            ix = np.argmax(yd)
            x=xd[ix]
            y=yd[ix]
            self.tx[i].set_position((x,y))
            self.tx[i].set_text("x:{}\ny:{:.3f}".format(x,y))
            self.marker[i].set_data([x],[y])
        return self
    def findmin(self):
        for i, line in enumerate(self.lines):
            xd,yd = line.get_data()
            ix = np.argmin(yd)
            x=xd[ix]
            y=yd[ix]
            self.tx[i].set_position((x,y))
            self.tx[i].set_text("x:{}\ny:{:.3f}".format(x,y))
            self.marker[i].set_data([x],[y])
        return self
    
#dm = DraggableMarker()
#
#plt.show()