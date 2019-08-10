#-*-coding:utf-8-*-

from builtins import str
from builtins import range
from builtins import object
import numpy as np
#from scipy import array,zeros,ones,asarray,eye,matrix,real,imag,angle,sqrt
from scipy import interpolate,linalg
from scipy import cos,sin,pi,log10
from scipy import optimize
# from string import upper,join
from numpy.lib.scimath import sqrt as csqrt
from copy import deepcopy
import network
from genel import *
import inspect

#program frekans birimi MHz
fcoef={"HZ":1.0, "KHZ":1e3, "MHZ":1e6, "GHZ":1e9}


def tukey_window(alpha,N):
    """
    Tukey window (also known as "tapered cosine window")
    """
    sonuc=[]
    for i in range(N):
        if (i<=alpha*(N-1.0)/2.0):
            sonuc.append(0.5*(1.0+cos(pi*(2.0*i/alpha/(N-1.0)-1.0))))
        elif (i<=(N-1)*(1.0-alpha/2.0)):
            sonuc.append(1.0)
        elif (i<=(N-1)):
            sonuc.append(0.5*(1.0+cos(pi*(2.0*i/alpha/(N-1.0)-2.0/alpha+1.0))))
    return sonuc

def gaussian_window(sigma,N):
    """
    Gaussian window
    sigma should be smaller than or equal to 0.5

    Ref: Wikipedia
    """
    sonuc=[]
    for i in range(N):
        sonuc.append(exp(-0.5*((i-(N-1.0)/2.0)/(sigma*(N-1.0)/2.0))**2))
    return sonuc
    
def cmp(x, y):
    """
    Replacement for built-in function cmp that was removed in Python 3

    Compare the two objects x and y and return an integer according to
    the outcome. The return value is negative if x < y, zero if x == y
    and strictly positive if x > y.
    """
    return int(x > y) - int(x < y)
    
def GenerateMultiportSPFile(conffile="", outputfilename=""):
    """
    Configuration file format:
    - comments start by "#"
    - every line's format is:
        i,j ? filename ? is, js
        meaning:
        S(is,js) of touchstone file filename is S(i,j) of outputfilename
    """
    dosya = open(conffile)
    target_indices = []
    source_indices = []
    spfilelist = []
    spfiles = {}
    for line in dosya:
        line = line.strip()
        if len(line) > 0:
            if line[0] != "#":
                a = line.split("?")
                target_indices.append(tuple([int(k) for k in a[0].strip().split(",")]))
                source_indices.append(tuple([int(k) for k in a[2].strip().split(",")]))
                filename = a[1].strip()
                if filename not in spfiles:
                    spfiles[filename] = spfile(filename)
                spfilelist.append(filename)

    dosya.close()
    temp = []
    for i in [list(k) for k in target_indices]:
        temp = temp + i
    port_sayisi = max(temp)
    # fill missing target_indices using reciprocity
    for i in range(1,port_sayisi+1):
        for j in range(1,port_sayisi+1):
            if ((i, j) not in target_indices) and ((j, i) in target_indices):
                print("i="+str(i),"j="+str(j))
                ind = target_indices.index((j, i))
                target_indices.append((i, j))
                source_indices.append(source_indices[ind])
                spfilelist.append(spfilelist[ind])

    newspfile = spfile(noktasayisi=len(spfiles[spfilelist[0]].getfrequencylist()), portsayisi=port_sayisi)
    newspfile.copyFrequencyFromSPFile(spfiles[spfilelist[0]])
    for i in range(len(target_indices)):
        print(target_indices[i][0], target_indices[i][1], source_indices[i][0], source_indices[i][1], spfilelist[i])
        newspfile.copyDataFromSPFile(target_indices[i][0], target_indices[i][1], source_indices[i][0],
                                     source_indices[i][1], spfiles[spfilelist[i]])
    newspfile.WriteToFile(outputfilename)

class spfile(object):
    """ Class to process Touchstone files.

    \todo
    1-Deembeding method

    """
    def __init__(self,dosya="",noktasayisi=0,portsayisi=0,satiratla=0):
        self.format="DB"
        self.frekans_birimi="HZ"
        self.empedans=[]
        self.dosya_adi=""
        self.sdata=[]
        self.ydata=[]
        self.zdata=[]
        self.abcddata=[]
        self.YZ_OK=0
        self.ABCD_OK=0
        self.formulation=1  # 1: "power-wave"
                            # 2: "pseudo-wave"
                            # 3: "HFSS pseudo-wave"
        self.comments=[] # format satırından önceki yorumlar
        if not dosya=="":
            self.dosyaoku(dosya,satiratla)
        else:
            self.empedans=[50.0]
            self.normalized=1 # normalized to 50 ohm if 1
            self.sdata=np.zeros((noktasayisi,portsayisi**2),complex)
            self.FrequencyPoints=np.zeros(noktasayisi,float)
            self.port_sayisi=portsayisi
            self.nokta_sayisi=noktasayisi
    def set_formulation(self, formulation):
        self.formulation = formulation
    def get_formulation(self):
        return self.formulation
    def new_file(self,frequency_list,data,format,port_sayisi,frekans_birimi="HZ"):
        self.FrequencyPoints=np.array(frequency_list)
        self.format=format
        if np.shape(data) is (len(frequency_list),port_sayisi):
            self.sdata=data
        else:
            self.sdata=np.zeros((len(frequency_list),port_sayisi))
        self.empedans=[50.0]
        self.nokta_sayisi=len(frequency_list)
        self.port_sayisi=port_sayisi
        self.frekans_birimi=frekans_birimi

    def copyDataFromSPFile(self,local_i,local_j,source_i,source_j,sourcespfile):
        """ Hey Dude!!!
        """
        local_column=(local_i-1)*self.port_sayisi+(local_j-1)
        source_column=(source_i-1)*self.port_sayisi+(source_j-1)
        self.sdata[:,local_column]=sourcespfile.data_array(format="COMPLEX",i=source_i,j=source_j)

    def copyFrequencyFromSPFile(self,sourcespfile):
        self.FrequencyPoints=sourcespfile.getfrequencylist()

    def getfilename(self):
        return self.dosya_adi

    def getformat(self):
        return self.format

    def setformat(self,format):
        self.format=format

    def setdatapoint(self,m,n,x):
        """
        m:  Frekans indeksi
        n:  Parametre indeksi
        x:  Deger (genelde kompleks bir sayi olmasi gerekir)
        """
        self.sdata[m,n]=x

    def setdatapoint2(self,m, indices,x):
        """
        m:  Frekans baslangic indeksi
        i,j:  Parametre indeksleri
        x:  Deger dizisi (genelde kompleks bir dizi olmasi gerekir)
        """
        (i,j) = indices
        for k in range(len(x)):
            self.sdata[k+m,(i-1)*self.port_sayisi+(j-1)]=x[k]

    def column_of_data(self,i,j):
        """Column of sdata that corresponds to Sij"""
        return (i-1)*self.port_sayisi+(j-1)

    def setdatapoint3(self, m, indices,x):
        """
        m:  frequency index
        indices: tuple(i,j)  Parameter indexes
        x:  Value (generally complex)
        """
        (i,j) = indices
        for k in range(len(x)):
            self.sdata[k+m,(i-1)*self.port_sayisi+(j-1)]=x[k]

    def SetSMatrixAtAFrequencyPoint(self,m,smatrix):
        """
        m: frequency index
        smatrix: S-Parameter matrix at m in np.matrix format
        """
        c=np.shape(smatrix)[0]
        smatrix.shape=1,c*c
        self.sdata[m,:]=np.array(smatrix)[0]

    def snp2smp(self,ports):
        """
        m: frequency index
        smatrix: S-Parameter matrix at m in np.matrix format
        """
        c=np.shape(smatrix)[0]
        smatrix.shape=1,c*c
        self.sdata[m,:]=np.array(smatrix)[0]

        """
        This method changes the port numbering of the network
        port j of new network corresponds to ports[j] in old network
        """
        ps=self.port_sayisi
        newps=len(ports)
        sdata=self.sdata # to speed-up
        new_sdata=np.zeros([self.nokta_sayisi,newps*newps]).astype(complex)
        for i in range(newps):
            for j in range(newps):
                n=(i)*ps+(j)
                m=(ports[i]-1)*ps+(ports[j]-1)
                new_sdata[:,n]=sdata[:,m]
        self.sdata=deepcopy(new_sdata)
        self.port_sayisi=newps
        self.empedans=[self.empedans[x] for x in ports]
        self.YZ_OK=0
        self.ABCD_OK=0

    def scaledata(self,scale=1.0, dataindices=[]):
        if (len(dataindices)==0):
            for i in range(self.port_sayisi**2):
                self.sdata[:,i]=self.sdata[:,i]*scale

    def get_no_of_ports(self):
        return self.port_sayisi

    def dosyayi_tekrar_oku(self):
        self.dosyaoku(self.dosya_adi)

    def dosyaoku(self,dosya_adi,satiratla=0):
        # print "satiratla ",satiratla
        self.dosya_adi=dosya_adi
        ext=dosya_adi.split(".")[-1]
        if ext[-1].lower()=="p":
            try:
                ps=self.port_sayisi=int(ext[-2])
            except:
                print("Error determining port number\n")
                return
        else: # for port numbers between 10-99 for which the extension is .s23 for example
            try:
                ps=self.port_sayisi=int(ext[-2])*10+int(ext[-1])
            except:
                print("Error determining port number\n")
                return

        try:
            f=open(dosya_adi,'r');
        except:
            print("Error opening the file: "+dosya_adi+"\n")
            return  0
        lines=f.readlines()[satiratla:]
        lsonuc=[]
        lfrekans=[]
        lines=[x.split("!")[0].strip() for x in lines if (len(x.strip())>0)]
        lines=[x for x in lines if len(x)>0]
        # bu durumda ilk satir # ile baslamali.
        x=lines[0]
        self.format,self.frekans_birimi,empedans=formatoku(x)
        self.empedans=np.ones(ps,dtype=complex)*empedans
        lines.remove(x)
        datalar=np.array((" ".join(lines)).split(),dtype=float)
        k=(2*ps**2+1)
        b=len(datalar)//k # self.nokta_sayisi
        datalar=datalar[:(b*k)]
        datalar.shape=b,k
        lfrekans=datalar[:,0]
        
        # frequencies should increase monotonically
        c=[1+cmp(datalar[i,0],datalar[i+1,0]) for i in range(b-1)]
        no_of_points=b
        try:
            no_of_points=c.index(1)
        except:
            pass
        try:
            no_of_points=min(no_of_points,c.index(2))
        except:
            pass
        b=no_of_points

        lfrekans=datalar[:b,0]*fcoef[self.frekans_birimi]

        if self.format=="RI":
            lsonuc=[datalar[:b,2*i+1]+datalar[:b,2*i+2]*1.0j for i in range(ps**2)]
        elif self.format=="DB":
            lsonuc=[10**((datalar[:b,2*i+1]/20.0))*cos(datalar[:b,2*i+2]*pi/180)+10**((datalar[:b,2*i+1]/20.0))*sin(datalar[:b,2*i+2]*pi/180)*1j for i in range(ps**2)]
        else:
            lsonuc=[datalar[:b,2*i+1]*cos(datalar[:b,2*i+2]*pi/180)+datalar[:b,2*i+1]*sin(datalar[:b,2*i+2]*pi/180)*1.0j for i in range(ps**2)]
        f.close()
        sdata=np.array(lsonuc,dtype=complex)
        sdata.resize(ps**2,b)
        sdata=sdata.T
        if ps==2:
            temp=deepcopy(sdata[:,2])
            sdata[:,2]=sdata[:,1]
            sdata[:,1]=deepcopy(temp)
        self.sdata=sdata
        self.FrequencyPoints=np.array(lfrekans)
        self.nokta_sayisi=b
        print(self.empedans)
        return 1

    def Ffunc(self,imp):
        """ Coefficient F in a, b definition of S-Parameters"""
        if self.formulation == 1:
            F=np.matrix(np.diag(np.sqrt((0.25/abs(imp.real)))))
        elif self.formulation == 2:
            F=np.matrix(np.diag(np.sqrt(abs(imp.real))/2/abs(imp)))
        elif self.formulation == 3:
            F=np.matrix(np.diag(np.sqrt(0.25/imp)))
        return F
        
    def CalcSYZ(self,input="S"):
        """ This function generate 2 of S, Y and Z parameters by the remaining parameter given.
        Y and Z-matrices calculated separately instead of calculating one and taking inverse. Because one of them may be undefined for some circuits.
        """
        # if self.YZ_OK==1:
            # return
        # if (self.YZ_OK==0 and (input=="Y" or input=="Z")):
            # print("Since Y and Z are not calculated before, input cannot be Y or Z")
            # return
        print(self.empedans)
        imp=self.PrepareRefImpedanceArray(self.empedans)
        impT=imp.T
        ps=self.port_sayisi
        ns=len(self.FrequencyPoints)
        birim=np.matrix(np.eye(ps))
        G=np.matrix(np.zeros((ps,ps),dtype=complex))
        F=np.matrix(np.zeros((ps,ps),dtype=complex))
        if input=="S":
            zdata=np.ones((ns,ps**2),dtype=complex)
            ydata=np.ones((ns,ps**2),dtype=complex)
            sdata=self.sdata
            for i in range(ns):
                G=np.matrix(np.diag(impT[:][i]))
                F=self.Ffunc(impT[:][i])
                smatrix = np.matrix(sdata[i,:]).reshape(ps,ps)
                try:
                    if self.formulation == 1:
                        ymatrix=F.I*(smatrix*G+G.conj()).I*(birim-smatrix)*F
                    else:
                        ymatrix=F.I*(smatrix*G+G  ).I*(birim-smatrix)*F 
                except:
                    print("Y-Matrix is undefined at frequency: {: f}\n".format(self.FrequencyPoints[i]))
                    break
                    
                try:
                    if self.formulation == 1:
                        zmatrix=F.I*(birim-smatrix).I*(smatrix*G+G.conj())*F
                    else:
                        zmatrix=F.I*(birim-smatrix).I*(smatrix*G+G)*F
                except:
                    print("Z-Matrix is undefined at frequency: {: f}\n".format(self.FrequencyPoints[i]))
                    break    

                zdata[i,:]=zmatrix.reshape(ps**2)
                ydata[i,:]=ymatrix.reshape(ps**2)
            self.ydata=ydata
            self.zdata=zdata
            self.YZ_OK=1
        elif input=="Z":
            zdata=self.zdata
            ydata=np.ones((ns,ps**2),dtype=complex)
            sdata=np.ones((ns,ps**2),dtype=complex)
            for i in range(ns):
                G=np.matrix(np.diag(impT[:][i]))
                F=self.Ffunc(impT[:][i])
                zmatrix=np.matrix(zdata[i,:]).reshape(ps,ps)
                try:
                    ymatrix=zmatrix.I
                    ydata[i,:]=ymatrix.reshape(ps**2)
                except:
                    print("Y-Matrix is undefined at frequency: {: f}\n".format(self.FrequencyPoints[i]))
                if self.formulation == 1:
                    smatrix=F*(zmatrix-G.conj())*(zmatrix+G).I*F.I
                else:
                    smatrix=F*(zmatrix-G)*(zmatrix+G).I*F.I
                sdata[i,:]=smatrix.reshape(ps**2)                
            self.ydata=ydata
            self.sdata=sdata
        elif input=="Y":
            ydata=self.ydata
            zdata=np.ones((ns,ps**2),dtype=complex)
            sdata=np.ones((ns,ps**2),dtype=complex)
            for i in range(ns):
                G=np.matrix(np.diag(impT[:][i]))
                F=self.Ffunc(impT[:][i])
                ymatrix=np.matrix(ydata[i,:]).reshape(ps,ps)
                try:
                    zmatrix=ymatrix.I
                    zdata[i,:]=zmatrix.reshape(ps**2)
                except:
                    print("Z-Matrix is undefined at frequency: {: f}\n".format(self.FrequencyPoints[i]))
                if self.formulation == 1:
                    smatrix=F*(zmatrix-G.H)*(zmatrix+G).I*F.I
                else:
                    smatrix=F*(zmatrix-G)*(zmatrix+G).I*F.I
                sdata[i,:]=smatrix.reshape(ps**2)
            self.zdata=zdata
            self.sdata=sdata

    def CalcTeigs(self,port1=1,port2=2):
        #Power Wave Formulation
        # if self.YZ_OK==1:
            # return
        self.S2ABCD(port1,port2)
        for i in range(len(self.FrequencyPoints)):
            abcd=self.abcddata[i].reshape(2,2)
            T=ABCD2T(abcd,[50.0+0j,50.0+0j])
            eigs,eigv=linalg.eig(T)
            print(self.FrequencyPoints[i],eigs)

    def S2ABCD(self,port1=1,port2=2):
        """ S-Matrix to ABCD matrix conversion between 2 chosen ports. Other ports are terminated with reference impedances
        """
        # if self.ABCD_OK==1:
            # return
        ns=len(self.FrequencyPoints)
        imp=self.PrepareRefImpedanceArray(self.empedans)
        impT=imp.T
        abcddata=np.ones((ns,4),dtype=complex)
        sdata=np.ones((ns,4),dtype=complex)
        sdata[:,0]=self.sdata[:,self.column_of_data(port1,port1)]
        sdata[:,1]=self.sdata[:,self.column_of_data(port1,port2)]
        sdata[:,2]=self.sdata[:,self.column_of_data(port2,port1)]
        sdata[:,3]=self.sdata[:,self.column_of_data(port2,port2)]
        birim=np.matrix(np.eye(2))
        for i in range(ns):
            smatrix=np.matrix(sdata[i,:]).reshape(2,2)
            F=self.Ffunc(impT[:][i])
            G=np.matrix(np.diag(impT[:][i]))
            if np.abs(np.linalg.det(birim-smatrix))<1e-10: # for circuits with undefined Z-matrix
                if self.formulation == 1:
                    ymatrix=F.I*(smatrix*G+G.conj()).I*(birim-smatrix)*F
                else:
                    ymatrix=F.I*(smatrix*G+G  ).I*(birim-smatrix)*F
                ABCD = network.Y2ABCD(ymatrix)
            else:
                if self.formulation == 1:
                    zmatrix=F.I*(birim-smatrix).I*(smatrix*G+G.conj())*F
                else:
                    zmatrix=F.I*(birim-smatrix).I*(smatrix*G+G)*F
                ABCD = network.Z2ABCD(zmatrix)
            abcddata[i,:]=ABCD.reshape(4)
        self.abcddata = abcddata
        # self.ABCD_OK=1
        return abcddata

    def Gmax(self,port1=1,port2=2):
        """ Maximum Transducer gain
        """
        self.S2ABCD(port1,port2)
        ns=len(self.FrequencyPoints)
        gmax=[]
        for i in range(ns):
            ABCD = self.abcddata[i,:].reshape(2,2)
            St=network.ABCD2S(ABCD,50.0).reshape(4)
            s11, s12, s21, s22 = tuple(flatten(St.tolist()))
            D=s11*s22-s12*s21
            K=(1-np.abs(s11)**2-np.abs(s22)**2+np.abs(D)**2)/np.abs(2*s12*s21)
            g=np.abs(s21)/np.abs(s12)*(K-csqrt(K*K-1))
            gmax=gmax+[g]
        return np.array(gmax)
        
    def Gav(self,port1=1,port2=2):
        """ Available gain from port1 to port2: Pav_toload/Pav_fromsource
        """
        self.S2ABCD(port1,port2)
        ns=len(self.FrequencyPoints)
        imp=self.PrepareRefImpedanceArray(self.empedans)
        ZL = imp[port2-1] 
        ZS = imp[port1-1] 
        GS=(ZS-50.0)/(ZS+50.0)
        GL=(ZL-50.0)/(ZL+50.0)
        gav=[]
        for i in range(ns):
            ABCD = self.abcddata[i,:].reshape(2,2)
            St=network.ABCD2S(ABCD,50.0).reshape(4)
            s11, s12, s21, s22 = tuple(flatten(St.tolist()))
            Gout=s22+s12*s21*GS[i]/(1-s11*GS[i])
            g=(1-np.abs(GS[i])**2)/np.abs(1.-s11*GS[i])**2*np.abs(s21)**2/(1-np.abs(Gout)**2)
            gav=gav+[g]
        return np.array(gav)
    
    def Gt(self,port1=1,port2=2):
        """ Transducer gain: Pload/Pav_fromsource
        """
        self.S2ABCD(port1,port2)
        ns=len(self.FrequencyPoints)
        imp=self.PrepareRefImpedanceArray(self.empedans)
        ZL = np.array(imp[port2-1] )
        ZS = np.array(imp[port1-1] )
        GS=(ZS-50.0)/(ZS+50.0)
        GL=(ZL-50.0)/(ZL+50.0)
        gt=[]
        for i in range(ns):
            ABCD = self.abcddata[i,:].reshape(2,2)
            St=network.ABCD2S(ABCD,50.0).reshape(4)
            s11, s12, s21, s22 = tuple(flatten(St.tolist()))
            Gout=s22+s12*s21*GS[i]/(1-s11*GS[i])
            g=(1-np.abs(GS[i])**2)/np.abs(1.-s11*GS[i])**2*np.abs(s21)**2*(1-np.abs(GL[i])**2)/np.abs(1-Gout*GL[i])**2
            gt=gt+[g]
        return np.array(gt)
    
    def StabilityFactor_K(self,frekanslar=[]):
        """ Calculates K factor """
        K=np.ones((len(self.FrequencyPoints),1),dtype=float)
        sdata=self.sdata
        for i in range(len(self.FrequencyPoints)):
            s11,s12,s21,s22=tuple(sdata[i,:])
            d=s11*s22-s12*s21
            K[i]=((1-abs(s11)**2-abs(s22)**2+abs(d)**2)/(2*abs(s21*s12)))
        return self.interpolate_data(self.FrequencyPoints, frekanslar, K)
            
    def interpolate_data(freqs1, freqs2, data):
        if freqs2 == []:
            return data
        tck_db = interpolate.splrep(freqs1,data,s=0,k=1)  #s=0, smoothing off, k=1, order of spline
        newdata = interpolate.splev(freqs2,tck_db,der=0)
        return newdata

    def StabilityFactor_mu1(self,frekanslar=[]):
        """ Calculates mu1 factor,
        Reference: High efficiency RF and Microwave Solid State Power Amplifiers, p55.
        """
        mu1=np.ones((len(self.FrequencyPoints),1),dtype=float)
        sdata=self.sdata
        for i in range(len(self.FrequencyPoints)):
            s11,s12,s21,s22=tuple(sdata[i,:])
            d=s11*s22-s12*s21
            mu1[i]=(1.0-abs(s11)**2)/(abs(s22-d*s11.conjugate())+abs(s21*s12))
        return self.interpolate_data(self.FrequencyPoints, frekanslar, mu1)

    def StabilityFactor_mu2(self,frekanslar=[]):
        """ Calculates mu2 factor,
        Reference: High efficiency RF and Microwave Solid State Power Amplifiers, p55.
        """
        mu2=np.ones((len(self.FrequencyPoints),1),dtype=float)
        sdata=self.sdata
        for i in range(len(self.FrequencyPoints)):
            s11,s12,s21,s22=tuple(sdata[i,:])
            d=s11*s22-s12*s21
            mu2[i]=(1.0-abs(s22)**2)/(abs(s11-d*s22.conj())+abs(s21*s12))
        return self.interpolate_data(self.FrequencyPoints, frekanslar, mu2)
 
    def changeRefImpedance(self,newimp):
        """ Changes reference impedance. Firstly impedance-independent Z-matrix is calculated and then S is calculated with new Zref using Z-matrix """
        self.CalcSYZ("S")
        for i in range(len(newimp)):
            if newimp[i]==None:
                newimp[i]=self.empedans[i]
        self.empedans=newimp
        self.CalcSYZ("Z")

    def PrepareRefImpedanceArray(self,imparray):
        """   Turns reference impedance array which is composed of numbers,arrays, functions and 1-ports to numerical array which
        is composed of numbers and arrays. It is made sure that abs(Re(Z))>0.
        """
        newarray=[]
        for i in range(self.port_sayisi):
            print("i= "+str(i)+" "+str(len(imparray)))
            if isinstance(imparray[i],spfile):
                newarray.append([x+(x.real==0)*1e-8 for x in imparray[i].data_array(format="COMPLEX",syz="Z",i=1,j=1, frekanslar=self.FrequencyPoints) ])
            elif inspect.isfunction(imparray[i]):
                newarray.append([x+(x.real==0)*1e-8 for x in imparray[i](self.FrequencyPoints)])
            elif isinstance(imparray[i],(float,complex)):
                newarray.append(np.ones(len(self.FrequencyPoints))*(imparray[i]+(imparray[i].real==0)*1e-8))
        return np.array(newarray)

    def ImpulseResponse(self,i=2,j=1,dcinterp=1,dcvalue=(0.0,0.0),MaxTimeStep=1.0,FreqResCoef=1.0):
    #   Determine the number of frequency steps
        nn=int(FreqResCoef*self.FrequencyPoints[-1]/(self.FrequencyPoints[-1]-self.FrequencyPoints[0])*len(self.FrequencyPoints)) #data en az kac noktada temsil edilecek
        df=(self.FrequencyPoints[-1]/(nn-0.5))   #frekans adimi
    #   Extrapolate to DC (append dc value and leave the rest to interpolation in the data_array method)
    #   Take data in frequency domain
        nfdata=linspace((df/2.0),self.FrequencyPoints[-1],nn) #frekans noktalari- df genisligindek bloklari orta noktalari
        adata=self.data_array(format="COMPLEX",syz="S",i=i,j=j, frekanslar=nfdata,DCInt=dcinterp,DCValue=dcvalue)
    #   Handle negative frequencies, Re(-w)=Re(w),Im(-w)=-Im(w), and prepare data array for ifft
        N=2**(int((log10(nn)/log10(2.0)))+10)
        data=np.zeros(N,dtype=complex)
        data[:nn]=deepcopy(adata)
        data[(N-nn):(N)]=deepcopy([x.conj() for x in adata[::-1]])
    #   Extrapolate to upper frequencies (Best to avoid)
    #   Windowing
        window=tukey_window(0.5,2*nn)
        datawindow=np.zeros(N,dtype=complex)
        datawindow[:nn]=deepcopy(window[nn:])
        datawindow[N-nn:]=deepcopy(window[:nn])
    #   IFFT
        F_data=np.fft.ifft(data*datawindow)
    #   Determine time step
        dt=1./N/df
    #   Generate Time Axis
        timeline=linspace(0,dt*(N-1),N)
        return (data,adata,datawindow,timeline,F_data)

    def __sub__(self,c):
        if (self.port_sayisi!=2 or c.port_sayisi!=2):
            print("Both networks should be two-port")
            return 0
        sonuc=deepcopy(self)
        sonuc.S2ABCD()
        c.ChangeFrequencyPoints(sonuc.FrequencyPoints)
        c.S2ABCD()
        for i in range(len(sonuc.FrequencyPoints)):
            abcd1=np.matrix(sonuc.abcddata[i].reshape(2,2))
            abcd2=np.matrix(c.abcddata[i].reshape(2,2))
            abcd=abcd1*abcd2.I
            s=network.ABCD2S(abcd,sonuc.empedans[0])
            sonuc.abcddata[i]=abcd.reshape(4)
            sonuc.sdata[i]=s.reshape(4)
            sonuc.YZ_OK=0
        return sonuc

    def __add__(self,c):
        if (self.port_sayisi!=2 or c.port_sayisi!=2):
            print("Both networks should be two-port")
            return 0
        sonuc=deepcopy(self)
        sonuc.S2ABCD()
        if len(sonuc.FrequencyPoints)>len(c.FrequencyPoints):
            print("Uyari: ilk devrenin frekans nokta sayisi, 2. devreninkinden fazla")
        c.ChangeFrequencyPoints(sonuc.FrequencyPoints)
        c.S2ABCD()
        for i in range(len(sonuc.FrequencyPoints)):
            abcd1=np.matrix(sonuc.abcddata[i].reshape(2,2))
            abcd2=np.matrix(c.abcddata[i].reshape(2,2))
            abcd=abcd1*abcd2
            s=network.ABCD2S(abcd,sonuc.empedans[0])
            sonuc.abcddata[i]=abcd.reshape(4)
            sonuc.sdata[i]=s.reshape(4)
            sonuc.YZ_OK=0
        return sonuc

    def CheckPassivity(self):
        """ This method determines the frequencies and frequency indices at which the network is non-passive
        """
        frekanslar=[]; # frequency points at which the network is non-passive
        indices=[];    # frequency indices at which the network is non-passive
        ps=self.port_sayisi
        for i in range(len(self.FrequencyPoints)):
            smatrix=np.matrix(self.sdata[i,:]).reshape(ps,ps)
            tempmatrix=(np.eye(ps)).astype(complex)-smatrix.H*smatrix
            eigs,eigv=linalg.eig(tempmatrix)
            for x in range(ps):
                if (eigs[x].real < 0):
                    frekanslar.append(self.FrequencyPoints[i])
                    indices.append(i)
        frekanslar=self.FrequencyPoints(indices)
        return  indices,frekanslar

    def RestorePassivity(self):
        """ Bu metod S-parametre datasinin pasif olmadigi frekanslarda
        S-parametre datasina mumkun olan en kucuk degisikligi yaparak
        S-parametre datasini pasif hale getirir.
        Referans:
        Restoration of Passivity In S-parameter Data of Microwave Measurements.pdf
        """
        pas, indices=self.CheckPassivity()
        t=self.port_sayisi**2
        c=np.zeros(2*t,np.float32)
        ps=self.port_sayisi
        gecici=[]
        sdata=self.sdata
        gecici.append(np.matrix('[1,0;0,0]').astype(complex))
        gecici.append(np.matrix('[1j,0;0,0]').astype(complex))
        gecici.append(np.matrix('[0,1;0,0]').astype(complex))
        gecici.append(np.matrix('[0,1j;0,0]').astype(complex))
        gecici.append(np.matrix('[0,0;1,0]').astype(complex))
        gecici.append(np.matrix('[0,0;1j,0]').astype(complex))
        gecici.append(np.matrix('[0,0;0,1]').astype(complex))
        gecici.append(np.matrix('[0,0;0,1j]').astype(complex))
        xvar=[1.0 for y in range(2*t)]
        smatrix=np.matrix(np.eye(ps)).astype(complex)
        perturbation =np.zeros((ps,ps)).astype(complex)
        for index in indices:
            for y in range((ps)**2):
                p=sdata[index,y]
                p1=p.real
                p2=p.imag
                smatrix[ (y/ps) , y%ps]=p

            while (1):
                tempmatrix=np.matrix(np.eye(ps).astype(complex))-smatrix.H*smatrix
                eigs,eigv=linalg.eig(tempmatrix)
                eigsl,eigvl=linalg.eig(tempmatrix.T)
                eigvl=eigvl.conjugate()
                dizi=[i for i in range(len(eigs)) if eigs[i].real<0]
                if (len(dizi)==0):
                    break
                else:
                    v=np.asmatrix(eigvl[:,dizi[0]]).T
                    u=np.asmatrix( eigv[:,dizi[0]]).T
                    # eigenvalue'daki gerekli degisim miktari
                    coef=min([-eigs[dizi[0]].real+1e-7,0.01])
                for y in range(2*t):
                    # Makalenin 5 numarali formulunun sag tarafindaki ifadesinde
                    # dS matrisinin her elemaninin yanindaki katsayilari verir.
                    c[y]=((((v.T)*(-(smatrix.H)*gecici[y]-(gecici[y].H)*smatrix)*u)/(v.T*u)))[0,0].real

                def constraint1(x):
                    """ Eigenvalue'nun yeni degerinin pozitif olmasi icin bu deger >0 olmali"""
                    return -coef+sum([x[i]*c[i] for i in range(2*t)])
                def constraint1_der(x):
                    """ constraint1 fonksiyonunun turevi """
                    return c
                def func_for_minimize(x):
                    """ bu eger minimize edilmeli
                        yani S-matrixteki degisim gerekli minimum duzeyde olmali
                    """
                    return sum([y**2 for y in x])
                def func_for_minimize_der(x):
                    """ func_for_minimize fonksiyonunun turevi """
                    return 2*x
                cons=({ 'type'   :   'ineq',
                        'fun'    :   constraint1,
                        'jac'    :   constraint1_der },)
                from scipy.optimize import minimize
                res = minimize(func_for_minimize, xvar, jac = func_for_minimize_der,
                               constraints = cons,  method = 'SLSQP', options={'disp': False})
                #print "cons ",constraint1(res.x)
                #print "x ",res.x
                x = res.x
                for y in range(t):
                    perturbation[(y/ps),y%ps]=x[2*y]+x[2*y+1]*1.0j
                smatrix=smatrix+perturbation
                #tempmatrix=np.matrix(np.eye(ps).astype(complex))-smatrix.H*smatrix
                #eigs,eigv=linalg.eig(tempmatrix)
                #print "eigs_after_iter ",eigs
            for y in range((ps)**2):
                sdata[index,y]=smatrix[ (y/ps) , y%ps]

    def WriteToFile(self,yenidosya="",parameter="S",frekans_birimi="",format=""):
        """
        Bu metod yeni parametre dosyasi olusturur.
        Eger verilen dosya isminde port sayisina uygun bir uzanti yoksa, uygun uzanti eklenir.
        """
        if yenidosya=="":
            yenidosya = self.dosya_adi
        if frekans_birimi=="":
            frekans_birimi=self.frekans_birimi
        if format=="":
            format=self.format

        ext = "s"+str(self.port_sayisi)
        if len(ext)<3: ext=ext+"p"
        if yenidosya[-4:].lower() != "."+ext:
            yenidosya = yenidosya+"."+ext

        f=open(yenidosya,'w')
        f.write("# "+frekans_birimi+" "+parameter+" "+format+" R "+str(self.empedans[0].real)+"\n")
        frekanslar=self.FrequencyPoints
        if parameter!="S":
            self.CalcSYZ()
            if parameter=="Y":
                data1=self.ydata
            elif parameter=="Z":
                data1=self.zdata
        else:
            data1=self.sdata
        ps=self.port_sayisi
        data=deepcopy(data1)
        if ps==2:
            temp=deepcopy(data[:,2])
            data[:,2]=data[:,1]
            data[:,1]=deepcopy(temp)
        temp=(1./fcoef[frekans_birimi])
        max_params_per_line=4
        if ps==3:
            max_params_per_line=3
        if (format=="RI"):
            for x in range(len(frekanslar)):
                print(str(frekanslar[x]*temp)+"    ", end=' ', file=f)
                for j in range(ps**2):
                    print("%-12.12f    %-12.12f " % (np.real(data[x,j]),np.imag(data[x,j])), end=' ', file=f)
                    if ((j+1)%max_params_per_line==0 or ((j+1)%ps**2==0)):
                        print("\n", end=' ', file=f)
        elif (format=="MA"):
            for x in range(len(frekanslar)):
                print(str(frekanslar[x]*temp)+"    ", end=' ', file=f)
                for j in range(ps**2):
                    print("%-12.12f    %-12.12f " % (np.abs(data[x,j]),np.angle(data[x,j],deg=1)), end=' ', file=f)
                    if ((j+1)%max_params_per_line==0 or ((j+1)%ps**2==0)):
                        print("\n", end=' ', file=f)
        else:
            for x in range(len(frekanslar)):
                print(str(frekanslar[x]*temp)+"    ", end=' ', file=f)
                for j in range(ps**2):
                    print("%-12.12f    %-12.12f " % (20*log10(np.abs(data[x,j])),np.angle(data[x,j],deg=1)), end=' ', file=f)
                    if ((j+1)%max_params_per_line==0 or ((j+1)%ps**2==0)):
                        print("\n", end=' ', file=f)
        f.close()

    def getfrequencylist(self):
        return self.FrequencyPoints

    def PortRemapping(self,ports):
        """
        This method changes the port numbering of the network
        port j of new network corresponds to ports[j] in old network

        ***obsolete***
        This function is substituted by snp2smp
        """
        ps=self.port_sayisi
        b=self.nokta_sayisi
        sdata=self.sdata
        new_sdata=np.zeros([b,ps*ps]).astype(complex)
        for i in range(ps):
            for j in range(ps):
                n=(i)*ps+(j)
                m=(ports[i]-1)*ps+(ports[j]-1)
                new_sdata[:,n]=sdata[:,m]
        self.sdata=deepcopy(new_sdata)
        self.YZ_OK=0
        self.ABCD_OK=0

    def add_abs_noise(self,dbnoise=0.1,phasenoise=0.1 ):
        """
        This method adds random amplitude and phase noise to the s-parameter data
        """
        # phase noise : degree
        n=self.port_sayisi**2
        ynew=[]
        sdata=np.zeros((len(self.FrequencyPoints),n),dtype=complex)
        for j in range(n):
            ydb=np.array([20*log10(abs(self.sdata[k,j])) for k in range(len(self.FrequencyPoints))])
            yphase=np.array([np.angle(self.sdata[k,j],deg=1)  for k in range(len(self.FrequencyPoints))]) # degree
            ynew_db = ydb+dbnoise*np.random.randn(len(ydb))
            ynew_ph = yphase+phasenoise*np.random.randn(len(yphase))
            ynew_mag=10**((ynew_db/20.0))
            ynew=ynew_mag*(cos(ynew_ph*pi/180)+1.0j*sin(ynew_ph*pi/180))
            # print np.shape(ynew),np.shape(self.sdata)
            sdata[:,j]=ynew
        self.sdata=sdata
        self.YZ_OK=0
        self.ABCD_OK=0

    def Smooth(self,smoothing_length=5):
        """
        This method applies moving average smoothing to the s-parameter data
        """
        n=self.port_sayisi**2
        ynew=[]
        sdata=np.zeros((len(self.FrequencyPoints),n),dtype=complex)
        for j in range(n):
            ydb=np.array([20*log10(abs(self.sdata[k,j])) for k in range(len(self.FrequencyPoints))])
            yphase=np.array([np.angle(self.sdata[k,j],deg=1)  for k in range(len(self.FrequencyPoints))]) # degree
            ynew_db=smooth(ydb,window_len=smoothing_length,window='hanning')
            ynew_ph=smooth(yphase,window_len=smoothing_length,window='hanning')
            ynew_mag=10**((ynew_db/20.0))
            ynew=ynew_mag*(cos(ynew_ph*pi/180)+1.0j*sin(ynew_ph*pi/180))
            sdata[:,j]=ynew
        self.sdata=sdata
        self.ABCD_OK=0
        self.YZ_OK=0

    def data_array(self,format="DB",syz="S",i=1,j=1, frekanslar=[],ref=None, DCInt=0,DCValue=(0.0,0.0),Smoothing=0, InterpolationConstant=0):
        """
        Returns a data array
        -For VSWR calculation j is ignored and only i is used.
        """
        #DCValue in (dB,deg), aci bilgisi +/- durumlarini kurtarmak icin.
        if format=="K":
            return self.StabilityFactor_K(frekanslar,i,j)
        if format.upper()=="MU1":
            return self.StabilityFactor_mu1(frekanslar,i,j)
        if format.upper()=="MU2":
            return self.StabilityFactor_mu2(frekanslar,i,j)
        if format=="VSWR" and i!=j:
            j=i
            return

        if frekanslar==[]:
            frekanslar=self.FrequencyPoints
        if InterpolationConstant > 0:
            frekanslar=self.FrequencyPoints
            frekstep = frekanslar[1]-frekanslar[0]
            frekanslar = np.array(list(range((len(frekanslar)-1)*InterpolationConstant+1)))*frekstep/InterpolationConstant+frekanslar[0]
        x=self.FrequencyPoints
        lenx=len(x)
        dcdb=[]
        dcph=[]
        if DCInt==1:
            dcdb=[DCValue[0]]
            dcph=[DCValue[1]]
            x=np.append([0.0],x)
        FORMAT=str.upper(format)
        n=(i-1)*self.port_sayisi+(j-1)
        ynew=[]
        if (str.upper(syz)=="Y" or str.upper(syz)=="Z"):
            self.CalcSYZ("S")
        if str.upper(syz)=="S":
            ydb=dcdb+[20*log10(abs(self.sdata[k,n])) for k in range(lenx)]
            yph=np.unwrap(dcph+[np.angle(self.sdata[k,n],deg=0)  for k in range(lenx)])*180.0/pi
        elif str.upper(syz)=="Y":
            ydb=dcdb+[20*log10(abs(self.ydata[k,n])) for k in range(lenx)]
            yph=np.unwrap(dcph+[np.angle(self.ydata[k,n],deg=0)  for k in range(lenx)])*180.0/pi
        elif str.upper(syz)=="Z":
            ydb=dcdb+[20*log10(abs(self.zdata[k,n])) for k in range(lenx)]
            yph=np.unwrap(dcph+[np.angle(self.zdata[k,n],deg=0)  for k in range(lenx)])*180.0/pi
        elif str.upper(syz)=="ABCD":
            self.S2ABCD()
            ydb=dcdb+[20*log10(abs(self.abcddata[k,n])) for k in range(lenx)]
            yph=np.unwrap(dcph+[np.angle(self.abcddata[k,n],deg=0)  for k in range(lenx)])*180.0/pi

        if frekanslar!=[]:
            tck_db = interpolate.splrep(x,ydb,s=0,k=2)  # s=0, smoothing off, k=2, order of spline
            ynew_db = interpolate.splev(frekanslar,tck_db,der=0)
            tck_phase = interpolate.splrep(x,yph,s=0,k=3)
            ynew_ph = interpolate.splev(frekanslar,tck_phase,der=0)
            if not ref==None:
                ynew_db=ynew_db-ref.data_array("DB","S",i,j,frekanslar)
                ynew_ph=ynew_ph-ref.data_array("UNWRAPPEDPHASE","S",i,j,frekanslar)
        else:
            ynew_db=array(ydb)
            ynew_ph=array(yph)
        if Smoothing>0:
            if Smoothing>lenx-1:
                Smoothing=lenx-1
            ynew_db=smooth(ynew_db,window_len=Smoothing,window='hanning')
            ynew_ph=smooth(ynew_ph,window_len=Smoothing,window='hanning')
        if  FORMAT=="COMPLEX":
            if frekanslar==[]: # interpolasyon yok.
                ynew=self.sdata[:,n]
            else:
                ynew_mag=10**((ynew_db/20.0))
                ynew=ynew_mag*(cos(ynew_ph*pi/180.0)+1.0j*sin(ynew_ph*pi/180.0))
        elif FORMAT=="DB":
            ynew = ynew_db
        elif FORMAT=="MAG":
            ynew = 10**((ynew_db/20.0))
        elif FORMAT=="VSWR":
            mag = 10**((ynew_db/20.0))
            ynew=((1.0+mag)/(1.0-mag))
        elif FORMAT=="REAL":
            ynew1 = 10**((ynew_db/20.0))
            ynew=ynew1*cos(ynew_ph*pi/180.0)
        elif FORMAT=="IMAG":
            ynew1 = 10**((ynew_db/20.0))
            ynew=ynew1*sin(ynew_ph*pi/180.0)
        elif FORMAT=="PHASE":
            ynew = np.mod(ynew_ph,360.)
        elif FORMAT=="UNWRAPPEDPHASE":
            ynew = ynew_ph
        elif FORMAT=="GROUPDELAY":
            if str.upper(syz)!="S":
                y=np.unwrap([angle(self.sdata[k,n],deg=1)  for k in range(lenx)])
                tck_phase = interpolate.splrep(x,y,s=0)
            uphase = interpolate.splev(frekanslar,tck_phase,der=0)
            t=len(frekanslar)
            ynew=list(range(t))
            if not ref==None:
                uphase=uphase-ref.data_array("UNWRAPPEDPHASE","S",i,j,frekanslar)
            for k in range(1,t-1):
                ynew[k]=-(uphase[k+1]-uphase[k-1])/(frekanslar[k+1]-frekanslar[k-1])/360.0
            ynew[0]=-(uphase[1]-uphase[0])/(frekanslar[1]-frekanslar[0])/360.0
            ynew[t-1]=-(uphase[t-1]-uphase[t-2])/(frekanslar[t-1]-frekanslar[t-2])/360.0

        return ynew
    def S(self,i=1,j=1,format="DB"):
        return self.data_array(format,"S",i,j)
    def Z(self,i=1,j=1,format="DB"):
        return self.data_array(format,"Z",i,j)
    def SetFrequencyPoints(self,frekanslar):
        self.FrequencyPoints=frekanslar
        self.nokta_sayisi=len(self.FrequencyPoints)
    def ChangeFrequencyPoints1(self,fstart,fstop,NumberOfPoints):
        self.ChangeFrequencyPoints(frekanslar=np.linspace(fstart,fstop,NumberOfPoints))
    def ChangeFrequencyPoints(self,frekanslar=[],InterpolationType=0):
        if frekanslar==[]:
            return
        x=self.FrequencyPoints
        self.FrequencyPoints=frekanslar
        n=self.port_sayisi**2
        ynew=[]        
        sdata=np.zeros((len(frekanslar),n),dtype=complex)
        for j in range(n):
            #dB interpolation
            y=[20*log10(abs(self.sdata[k,j])) for k in range(len(x))]
            tck_db = interpolate.splrep(x,y,s=0)
            #phase interpolation
            y=[np.angle(self.sdata[k,j],deg=1)  for k in range(len(x))]
            tck_phase = interpolate.splrep(x,y,s=0)
            ynew_db = interpolate.splev(frekanslar,tck_db,der=0)
            ynew_ph = interpolate.splev(frekanslar,tck_phase,der=0)
            ynew_mag=10**((ynew_db/20.0))
            ynew=ynew_mag*(cos(ynew_ph*pi/180)+1.0j*sin(ynew_ph*pi/180))
            sdata[:,j]=ynew
        self.sdata=sdata
        self.nokta_sayisi=len(self.FrequencyPoints)
    def Convert_S1P_to_S2P(self):
        #thru kalibrasyonla S21 olcumu  durumu icin, pasif devreler icin.
        self.port_sayisi=2
        temp=(self.port_sayisi)**2
        newdata=np.zeros([self.nokta_sayisi,temp]).astype(complex)
        newdata[:,0]=1e-6+0j
        newdata[:,1]=self.sdata[:,0]
        newdata[:,2]=self.sdata[:,0]
        newdata[:,3]=1e-6+0j
        self.sdata=newdata
def  formatoku(cumle):
    cumle=str.upper(cumle).strip()
    a=cumle[1:].split()
    format=a[2]
    frekans=a[0]
    if len(a)>3:
        empedans=float(a[4])
    else:
        empedans=50.0
    return format,frekans,empedans
def grafik(p1,p2,stil):
    plot(p1,p2,stil,hold='True')

if __name__ == '__main__':
    aa=spfile()
    aa.dosyaoku("./WG_EvanescentModeFilter Filtre_kapasiteli_loss_durumu1.s2p")
    bb=spfile()
    bb.dosyaoku("./WG_EvanescentModeFilter Filtre_kapasiteli_loss_durumu.s2p")
    # aa.dosyaoku("./5port_kapasiteli3.s5p")
    # a=0.016
    # b=0.0005
    # er=3.38
    # c1=0.564e-12
    # c2=0.836e-12
    # aa.changeRefImpedance_powerwave([lambda x:-0.5j/pi/x/c1*(1+0.01j),lambda x:-0.5j/pi/x/c2*(1+0.01j),lambda x:-0.5j/pi/x/c1*(1+0.01j),lambda x:Z_WG_TE10(er,a,b,x)*(1+0.01j),lambda x:Z_WG_TE10(er,a,b,x)*(1+0.01j)])
    (data,adata,window,timeline,F_data)=aa.ImpulseResponse(i=2,j=1,dcinterp=1,dcvalue=(-200.0,0.0),FreqResCoef=10.0)
    (bdata,badata,bwindow,btimeline,bF_data)=bb.ImpulseResponse(i=2,j=1,dcinterp=1,dcvalue=(-200.0,0.0),FreqResCoef=10.0)
    from pylab import *
    #plot(window)
    hold(True)
    #plot(abs(data)*window)
    plot(timeline[1:],abs(F_data[1:]))
    figure()
    plot(btimeline[1:],abs(bF_data[1:]))
    #plot(abs(bdata)*bwindow)
    #plot(20.0*log10(abs(np.fft.fft(F_data))))
    show()
