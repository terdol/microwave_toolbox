import numpy as np

def refinearray(inp):
    i=0
    if len(inp)<2:
        return inp
    while 1:
        if np.abs(inp[i]-inp[i+1])/np.sum(inp)<1e-5:
            inp.pop(i+1)
        else:
            i=i+1
        if (i>=len(inp)-1):
            break
    return inp

class subregion:
    def __init__(self):
        self.start=0
        self.end=0
        self.r=0
        self.mincell=0
        self.maxcell=0
        self.startindex=0
        self.endindex=0
        self.Ncell=0
        self.shape=0  #shape: uniform=1, FC=2, CF=3, FCF=4
        #startindex bolgenin icindeki ilk hucreyi isaret eder, endindex bolgenin disindaki ilk hucreyi isaret eder.

class CartesianGrid():
    def __init__(self):
        self.sizex=0
        self.sizey=0
        self.sizez=0
        self.sensitivity=1e-5
        self.number_of_abc_layers_x1=0
        self.number_of_abc_layers_x2=0
        self.number_of_abc_layers_y1=0
        self.number_of_abc_layers_y2=0
        self.number_of_abc_layers_z1=0
        self.number_of_abc_layers_z2=0
        self.dx=[]
        self.dy=[]
        self.dz=[]
        self.KritikNoktalar_x=[]
        self.KritikNoktalar_y=[]
        self.KritikNoktalar_z=[]
        self.delta=0
        self.deltamin=0
        self.unit ="m"
        self.maxratio=1.2

    def findsubregions(self, kritikx,kritiky,kritikz):

        kritikx.sort()
        kritiky.sort()
        kritikz.sort()

        refinearray(kritikx)
        refinearray(kritiky)
        refinearray(kritikz)

        self.KritikNoktalar_x=kritikx
        self.KritikNoktalar_y=kritiky
        self.KritikNoktalar_z=kritikz

        self.sizex=len(kritikx)
        self.sizey=len(kritiky)
        self.sizez=len(kritikz)

        subx = []
        suby = []
        subz = []

        for i in range(self.sizex-1):
            sr = subregion()
            sr.start = kritikx[i]
            sr.end = kritikx[i+1]
            sr.shape = 1
            subx.append(sr)

        for i in range(self.sizey-1):
            sr = subregion()
            sr.start = kritiky[i]
            sr.end = kritiky[i+1]
            sr.shape = 1
            suby.append(sr)

        for i in range(self.sizez-1):
            sr = subregion()
            sr.start = kritikz[i]
            sr.end = kritikz[i+1]
            sr.shape = 1
            subz.append(sr)

        self.subx=subx
        self.suby=suby
        self.subz=subz

    def optimizegrids(self):
        maxratio=self.maxratio
        sublists = [self.subx,self.suby,self.subz]
        gridler = [self.dx,self.dy,self.dz]
        policy=0
        for j in range(3):
            sublist=sublists[j]
            dd=gridler[j]
            if len(sublist)<2:
                return
            if dd[sublist[0].endindex-1]>maxratio*dd[sublist[0].endindex]:
                self.customgrid(maxratio*dd[sublist[0].endindex], dd[sublist[0].startindex], 0, maxratio, 1, j, policy)
            for i in range(1,len(sublist)-1):
                if dd[sublist[i].startindex]>maxratio*dd[sublist[i].startindex-1] and dd[sublist[i].endindex-1]>maxratio*dd[sublist[i].endindex]:
                    self.customgrid(maxratio*np.min((dd[sublist[i].startindex-1],dd[sublist[i].endindex])), dd[sublist[i].startindex], i, maxratio, 2, j, policy)
                elif dd[sublist[i].startindex]>maxratio*dd[sublist[i].startindex-1]:
                    self.customgrid(maxratio*dd[sublist[i].startindex-1], dd[sublist[i].endindex-1], i, maxratio, 0, j, policy)
                elif dd[sublist[i].endindex-1]>maxratio*dd[sublist[i].endindex]:
                    self.customgrid(maxratio*dd[sublist[i].endindex], maxratio*dd[sublist[i].startindex-1], i, maxratio, 1, j, policy)
            if dd[sublist[-1].startindex]>maxratio*dd[sublist[-1].startindex-1]:
                self.customgrid(maxratio*dd[sublist[-1].startindex-1], dd[sublist[-1].endindex-1], -1, maxratio, 0, j, policy)


    def nu_grid(self, deltax, deltay, deltaz):
        dx=[]
        dy=[]
        dz=[]
        sizex = self.sizex
        sizey = self.sizey
        sizez = self.sizez
        subx = self.subx
        suby = self.suby
        subz = self.subz
        temp=0
        for i in range(self.sizex-1):
            grids=self.ugrid(deltax,subx[i].end-subx[i].start)
            subx[i].Ncell=len(grids)
            subx[i].startindex=temp
            temp+=subx[i].Ncell
            subx[i].endindex=temp
            dx.extend(grids)
            subx[i].mincell=grids[0]
            subx[i].maxcell=grids[0]
        temp=0
        for i in range(self.sizey-1):
            grids=self.ugrid(deltay,suby[i].end-suby[i].start)
            suby[i].Ncell=len(grids)
            suby[i].startindex=temp
            temp+=suby[i].Ncell
            suby[i].endindex=temp
            dy.extend(grids)
            suby[i].mincell=grids[0]
            suby[i].maxcell=grids[0]
        temp=0
        for i in range(self.sizez-1):
            grids=self.ugrid(deltaz,subz[i].end-subz[i].start)
            subz[i].Ncell=len(grids)
            subz[i].startindex=temp
            temp+=subz[i].Ncell
            subz[i].endindex=temp
            dz.extend(grids)
            subz[i].mincell=grids[0]
            subz[i].maxcell=grids[0]
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.subx = subx
        self.suby = suby
        self.subz = subz


    def set_number_of_abc_layers(self, x1, x2, y1, y2, z1, z2):
        if (x1>=0): self.number_of_abc_layers_x1=x1
        if (x2>=0): self.number_of_abc_layers_x2=x2
        if (y1>=0): self.number_of_abc_layers_y1=y1
        if (y2>=0): self.number_of_abc_layers_y2=y2
        if (z1>=0): self.number_of_abc_layers_z1=z1
        if (z2>=0): self.number_of_abc_layers_z2=z2

    def writetofile(self, dosya):
        pass
    def getsubregion(self, i):
        x1=self.subx[i].start
        x2=self.subx[i].end
        y1=self.suby[i].start
        y2=self.suby[i].end
        z1=self.subz[i].start
        z2=self.subz[i].end
        return x1, x2, y1, y2, z1, z2

    def getsubregion_p(self, j, i):
        if (j==0):
            return (self.subx[i].start,self.subx[i].end)
        elif (j==1):
            return (self.suby[i].start,self.suby[i].end)
        elif (j==2):
            return (self.subz[i].start,self.subz[i].end)
        return (0,0)

    def customgrid(self, dmin, dmax, index, rmax, shape, axis, policy):
        """ index: subregion numarasi
            rmax =      maximum ratio between neighboring grids
            shape =      0- fine-coarse
                        1- coarse-fine
                        2- fine-coarse-fine
                        3- uniform
            policy =    0- minimum number of grids
                        1- minimum ratio, smoother transition
        """

        dx = self.dx
        dy = self.dy
        dz = self.dz
        sizex = self.sizex
        sizey = self.sizey
        sizez = self.sizez
        subx = self.subx
        suby = self.suby
        subz = self.subz
        dmin, dmax = np.min((dmin,dmax)), np.max((dmin,dmax))
        if (axis==0):
            if (index>=sizex-1):
                print("hata")
                return

            bas=subx[index].startindex
            son=subx[index].endindex
            length=subx[index].end-subx[index].start
            grids = self.nugrid(dmin,dmax,rmax,length,shape,policy)
            subx[index].Ncell=len(grids)
            subx[index].endindex=len(grids)+bas
            for i in range(index+1, sizex-1):
                subx[i].startindex+=len(grids)-(son-bas)
                subx[i].endindex+=len(grids)-(son-bas)

            del dx[bas:son]
            dx[bas:bas] = grids
            subx[index].mincell=grids[0]
            subx[index].maxcell=grids[len(grids)-1]

        elif (axis==1):
            if (index>=sizey-1):
                print("hata")
                return

            bas=suby[index].startindex
            son=suby[index].endindex
            length=suby[index].end-suby[index].start
            grids=self.nugrid(dmin,dmax,rmax,length,shape,policy)
            suby[index].Ncell=len(grids)
            suby[index].endindex=len(grids)+bas
            for i in range(index+1,sizey-1):
                suby[i].startindex+=len(grids)-(son-bas)
                suby[i].endindex+=len(grids)-(son-bas)

            del dy[bas:son]
            dy[bas:bas] = grids
            suby[index].mincell=grids[0]
            suby[index].maxcell=grids[len(grids)-1]


        elif (axis==2):
            if (index>=sizez-1):
                print("hata")
                return

            bas=subz[index].startindex
            son=subz[index].endindex
            length=subz[index].end-subz[index].start
            grids=self.nugrid(dmin,dmax,rmax,length,shape,policy)
            subz[index].Ncell=len(grids)
            subz[index].endindex=len(grids)+bas
            for i in range(index+1,sizez-1):
                subz[i].startindex+=len(grids)-(son-bas)
                subz[i].endindex+=len(grids)-(son-bas)

            del dz[bas:son]
            dz[bas:bas] = grids
            subz[index].mincell=grids[0]
            subz[index].maxcell=grids[len(grids)-1]

        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.subx = subx
        self.suby = suby
        self.subz = subz

    def reshapegrid(self, delta, index, rmax, shape, axis, policy):
        #index: subregion numarasi
        dx = self.dx
        dy = self.dy
        dz = self.dz
        sizex = self.sizex
        sizey = self.sizey
        sizez = self.sizez
        subx = self.subx
        suby = self.suby
        subz = self.subz
        tip=0
        if (axis==0):
            if (index==0):
                d1=delta
                d2=dx[subx[0].endindex]
            elif (index==(sizex-2)):
                d2=delta
                d1=dx[subx[sizex-2].startindex-1]
            elif ((index>0) and (index<(sizex-2))):
                d1=dx[subx[index].startindex-1]
                d2=dx[subx[index].endindex]
            else:
                print("Error in index number ")

        elif (axis==1):
            if (index==0):
                d1=delta
                d2=dy[suby[0].endindex]

            elif (index==(sizey-2)):
                d2=delta
                d1=dy[suby[sizey-2].startindex-1]

            elif ((index>0) and (index<(sizey-2))):
                d1=dy[suby[index].startindex-1]
                d2=dy[suby[index].endindex]

            else:
                print("Error in index number ")

        elif (axis==2):
            if (index==0):
                d1=delta
                d2=dz[subz[0].endindex]

            elif (index==(sizez-2)):
                d2=delta
                d1=dz[subz[sizez-2].startindex-1]

            elif ((index>0) and (index<(sizez-2))):
                d1=dz[subz[index].startindex-1]
                d2=dz[subz[index].endindex]

            else:
                print("Error in index number ")

        if (d2<d1):
            tip=1
        else:
            tip=0
        self.customgrid(d1,d2,index,rmax,tip,axis,policy)

    def sumz(self):
        return np.sum(dz)

    def get_grids_sub_i(self, i, axis):
        if (axis==0):
            return self.dx[self.subx[i].startindex:self.subx[i].endindex]
        elif (axis==1):
            return self.dy[self.suby[i].startindex:self.suby[i].endindex]
        elif (axis==2):
            return self.dz[self.subz[i].startindex:self.subz[i].endindex]

    def get_subregion_index_of_point(self, coor, axis):
        i=0
        if (axis==0):
            while (i<self.sizex):
                if ((coor>=self.subx[i].start) and (coor<=self.subx[i].end)):
                    return i
                else:
                    i+=1

        elif (axis==1):
            while (i<self.sizey):
                if ((coor>=self.suby[i].start) and (coor<=self.suby[i].end)):
                    return i
                else:
                    i+=1

        elif (axis==2):
            while (i<self.sizez):
                if ((coor>=self.subz[i].start) and (coor<=self.subz[i].end)):
                    return i
                else:
                    i+=1

    def loadfromfile(self, dosya):
        pass

    def subregionparams(self, c, grid):
        if (grid==0):
            a = self.subx[c].mincell
            b = self.subx[c].maxcell
            shape = self.subx[c].shape
        elif (grid==1):
            a = self.suby[c].mincell
            b = self.suby[c].maxcell
            shape = self.suby[c].shape
        elif (grid==2):
            a = self.subz[c].mincell
            b = self.subz[c].maxcell
            shape = self.subz[c].shape
        return a, b, shape

    def return_subregion_params(self, grid, index):
        if (grid==0):
            a = self.subx[index].mincell
            b = self.subx[index].maxcell
        elif (grid==1):
            a = self.suby[index].mincell
            b = self.suby[index].maxcell
        elif (grid==2):
            a = self.subz[index].mincell
            b = self.subz[index].maxcell
        return (a,b)

    def ugrid(self, delta, length, parite=0):
        N = int(np.ceil(length/delta))
        if ((parite==-1) and (N%2==0)):
            N = N+1
        if ((parite==1) and (N%2==1)):
            N = N+1
        if (N==0):
            N=1
        grids = [length/N]*N
        return grids

    def nugrid(self, delta1, delta2, rmax, length, shape,  policy=0,cc=0.99):
        """ rmax =      maximum ratio between neighboring grids
            length=     total length
            shape =      0- fine-coarse
                        1- coarse-fine
                        2- fine-coarse-fine
                        3- uniform
            policy =    0- minimum number of grids
                        1- minimum ratio, smoother transition
        """
        r=rmax

        if (shape==3):
            grids=self.ugrid(delta1,length)
        elif  (shape==2):
            grids=self.nugrid(delta1,delta2,rmax,length/2.0,0, policy)
            temp=self.nugrid(delta1,delta2,rmax,length/2.0,1, policy)
            grids = grids + temp
        elif  (shape==0 or shape==1):
            grids=[]
            deltamin=np.min([delta1,delta2])
            deltamax=np.max([delta1,delta2])
            if (policy==1):
                r=(length-deltamin)/(length-deltamax)
                N=round(float(np.ceil(np.log(deltamax/deltamin)/np.log(r))))
                while(1):
                    temp3=deltamin*(r**(N+1)-1)/(r-1)
                    if (np.abs(temp3/length-1)<self.sensitivity):
                        break
                    elif (temp3>length):
                        r=r*cc
                    elif (temp3<length):
                        r=r*(2.0-cc)
                error = length-temp3
                grids = grids + [deltamin*np.power(r,i) for i in range(N+1)]
                grids[-1] += error
            elif (policy==0):
                N=1+int(np.ceil(np.log(deltamax/deltamin)/np.log(rmax)))
                temp2=deltamin*(np.power(rmax,N)-1)/(rmax-1)
                if (temp2<length):
                    N3=int(np.ceil((length-temp2)/deltamax))
                    while(1):
                        temp3=deltamin*(np.power(r,N)-1)/(r-1)+N3*deltamin*np.power(r,N-1)
                        if (np.abs(temp3/length-1)<self.sensitivity):
                            break
                        elif (temp3>length):
                            r=r*cc
                        elif (temp3<length):
                            r=r*(2.0-cc)
                    error = length-temp3
                    grids = grids + [deltamin*np.power(r,i) for i in range(N)]
                    grids[-1] += error
                    grids = grids + [deltamin*np.power(r,N-1) for i in range(N3)]
                else:
                    N1=int(np.ceil(np.log(length*(rmax-1)/deltamin+1.)/np.log(rmax)))
                    while(1):
                        temp3=deltamin*(np.power(r,N1)-1)/(r-1)
                        if (np.abs(temp3/length-1)<self.sensitivity):
                            break
                        elif (temp3>length):
                            r=r*cc
                        elif (temp3<length):
                            r=r*(2.0-cc)
                    error = length-temp3
                    grids = grids + [deltamin*np.power(r,i) for i in range(N1)]
                    grids[-1] += error
            if (shape==1):
                grids.reverse()
        return grids

    def SetUnit(self,s):
        self.unit=s

    def get_nof_sub(self, eksen):
        if (eksen==0):
            return self.sizex-1
        elif (eksen==1):
            return self.sizey-1
        elif (eksen==2):
            return self.sizez-1
        return 0

if __name__ == "__main__":
    # krix=[0,1.0,1.05,2.0]
    # aa=CartesianGrid()
    # aa.findsubregions(krix,[],[])
    # aa.nu_grid(0.1,0.1,0.1)
    # aa.optimizegrids()
    # import matplotlib.pyplot as plt
    # plt.plot(aa.dx)
    # plt.show()
    cg = CartesianGrid()
    import time
    t1 = time.time()
    cg.nugrid(1.0,2.0,1.2,20,0, 0,cc=0.999)
    t2 = time.time()
    cg.nugrid(1.0,2.0,1.2,20,0, 0,cc=0.99)
    t3 = time.time()
    cg.nugrid(1.0,2.0,1.2,20,0, 0,cc=0.9)
    t4 = time.time()
    cg.nugrid(1.0,2.0,1.2,20,0, 0,cc=0.5)
    t5 = time.time()
    print("t2-t1= "+str(t2-t1))
    print("t3-t2= "+str(t3-t2))
    print("t4-t3= "+str(t4-t3))
    print("t5-t4= "+str(t5-t4))                   
    pass
