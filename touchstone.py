#-*-coding:utf-8-*-
# pragma pylint: disable=too-many-function-args
import numpy as np
from numpy.linalg import eig
try:
    import scipy.interpolate
    import scipy.optimize
except:
    pass

# from scipy.linalg import eig
# from scipy.signal import get_window
from numpy.lib.scimath import sqrt as csqrt
import sympy as sp
from copy import deepcopy
import network
import itertools
from genel import *
from myconstants import *
import inspect
from collections import defaultdict
import transmission_lines as tlines


fcoef={"HZ":1.0, "KHZ":1e3, "MHZ":1e6, "GHZ":1e9}

def parseformat(cumle):
    cumle=str.upper(cumle).strip()
    a=cumle[1:].strip().split()
    dataformat=a[2]
    frequency=a[0]
    if len(a)==3:
        refimpedance=None
    elif len(a)==5:
        refimpedance=float(a[4])
    return dataformat,frequency,refimpedance

def generate_multiport_spfile(conffile="", outputfilename=""):
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
                ind = target_indices.index((j, i))
                target_indices.append((i, j))
                source_indices.append(source_indices[ind])
                spfilelist.append(spfilelist[ind])

    # newspfile = spfile(noktasayisi=len(spfiles[spfilelist[0]].get_frequency_list()), portsayisi=port_sayisi)
    # newspfile.copy_frequency_from_spfile(spfiles[spfilelist[0]])
    newspfile = spfile(freqs=spfiles[spfilelist[0]].get_frequency_list(), portsayisi=port_sayisi)
    for i in range(len(target_indices)):
        newspfile.copy_data_from_spfile(target_indices[i][0], target_indices[i][1], source_indices[i][0],
                                     source_indices[i][1], spfiles[spfilelist[i]])
    newspfile.write2file(outputfilename)

def extract_gamma_ereff(file_long,file_short,dL,sm=1):
    """Extraction of complex propagation constant (gamma) and complex effective permittivity from the S-Parameters of 2 uniform transmission lines with different lengths.

    Args:
        file_long (str): S-Parameter filename of longer line.
        file_short (str): S-Parameter filename of shorter line.
        dL (float): Difference of lengths of two lines (positive in meter).
        sm (int, optional): If this is larger than 1, this is used as number of points for smoothing. Defaults to 1.

    Returns:
        tuple: tuple of two complex numpy arrays (gamma, er_eff).
    """
    sp1=spfile(file_short)
    sp2=spfile(file_long)
    if sm>1:
        sp1.smoothing(sm)
        sp2.smoothing(sm)

    freks=sp1.get_frequency_list()
    w=2*np.pi*freks

    net1=sp2-sp1
    net2=-sp1+sp2
    net1.s2abcd()
    net2.s2abcd()

    a1=np.arccosh(0.5*(net1.abcddata[:,0]+net1.abcddata[:,3]))
    a2=np.arccosh(0.5*(net2.abcddata[:,0]+net2.abcddata[:,3]))
    a1=np.real(a1)+np.unwrap(np.imag(a1))*1j
    a2=np.real(a2)+np.unwrap(np.imag(a2))*1j
    dA=0.5*(a1+a2)
    gamma=dA/dL
    gamma=np.abs(np.real(gamma))+1j*np.abs(np.imag(gamma))
    er_eff  = -(c0*gamma/w)**2
    return (gamma,er_eff)

def extract_gamma_ereff_all(files,Ls,sm=1):
    """Extraction of average complex propagation constant (gamma) and complex effective permittivity from the S-Parameters of multiple uniform transmission lines with different lengths.

    Args:
        files (list): List of S-Parameter filenames of transmission lines.
        Ls (list): List of lengths of transmission lines in the same order as *files* parameter.
        sm (int, optional): If this is larger than 1, this is used as number of points for smoothing. Defaults to 1.

    Returns:
        tuple: tuple of two complex numpy arrays (gamma, er_eff).
    """
    files_Ls=list(zip(files,Ls))
    files_Ls.sort(key=lambda x:-x[1])
    cifts = list(itertools.combinations(files_Ls,2))
    eeff_all=()
    gamma_all=()
    for cift in cifts:
        (gamma,eeff)=extract_gamma_ereff(cift[0][0],cift[1][0],cift[0][1]-cift[1][1],sm)
        eeff_all = eeff_all + (eeff,)
        gamma_all = gamma_all + (gamma,)
    gamma_av  = sum(gamma_all  )/len(cifts)
    eeff_av= sum(eeff_all)/len(cifts)
    return (gamma_av, eeff_av)

def cascade_2ports(filenames):
    if isinstance(filenames[0],str):
        first=spfile(filenames[0])
    else:
        first=deepcopy(filenames[0])
    for i in range(1,len(filenames)):
        if isinstance(filenames[i],str):
            sp=spfile(filenames[i])
        else:
            sp=filenames[i]
        first.connect_network_1_conn(sp,2,1)
    return first

def average_networks(networks):
    for i in range(len(networks)):
        if isinstance(networks[i],str):
            networks[i]=spfile(networks[i])
    # outputnetwork=deepcopy(networks[0])
    nop=networks[0].get_no_of_ports()
    freks=networks[0].get_frequency_list()
    outputnetwork = spfile(freqs=freks,portsayisi=nop)
    N = len(networks)
    sdatas = [net.sdata for net in networks]
    sdata=sum(sdatas)/N
    outputnetwork.sdata=sdata
    return outputnetwork

def untermination_method(g1,g2,g3,gL1,gL2,gL3,returnS2P=False, freqs=None):
    """Determination of :math:`S_{11}`, :math:`S_{22}` and :math:`S_{21}=S_{12}` for a 2-port network network using 3 reflection coefficient values at port-1 for 3 terminations at port-2. :math:`S_{21}` can only be calculated with a sign ambiguity because it exists only as square in the formulae.

    Port-1: Input port.
    Port-2: Output port where load impedances are switched.

    Args:
        g1 (float, complex or ndarray): Reflection coefficient at port-1 when port-2 is terminated by a load with reflection coefficient gL1
        g2 (float, complex or ndarray): Reflection coefficient at port-1 when port-2 is terminated by a load with reflection coefficient gL2
        g3 (float, complex or ndarray): Reflection coefficient at port-1 when port-2 is terminated by a load with reflection coefficient gL3
        gL1 (float, complex or ndarray): Reflection coefficient of load at port-2 that gives g1 reflection coefficient at port-1
        gL2 (float, complex or ndarray): Reflection coefficient of load at port-2 that gives g2 reflection coefficient at port-1
        gL3 (float, complex or ndarray): Reflection coefficient of load at port-2 that gives g3 reflection coefficient at port-1
        returnS2P (boolean): If True, function returns an *spfile* object of the 2-port network, if False, it returns 3-tuple of S-Parameter arrays. Default is False.
        freqs (numpy.ndarray, list): If returnS2P is True, this input is used as the frequency points of the returned *spfile* object. Default is None.

    Returns:
        tuple: Either 3-Element tuple of (S11, S22, S21) or *spfile* object, depending on returnS2P input.
    """
    # s22=a+b.S11
    a=(g1*gL2-g2*gL1)/(gL1*gL2)/(g1-g2)
    b=(gL1-gL2)/(gL1*gL2)/(g1-g2)
    gLall=gL1*gL2*gL3
    Y3 = b*g3*gLall+gL2*gL1
    Z3 = g3*gL2*gL1-a*g3*gLall
    Y1 = b*g1*gLall+gL2*gL3
    Z1 = g1*gL2*gL3-a*g1*gLall
    S11 = -(Z3-Z1)/(Y1-Y3)
    S22 = a+b*S11
    S21 = np.sqrt((g1-S11)*(1-S22*gL1)/gL1)
    if returnS2P==True:
        if freqs is None:
            freqs=np.linspace(1e9,10e9,len(g1))
        ph = np.unwrap(np.angle(S21,deg=0))
        for i in range(1,len(ph)):
            phi1=np.angle(S21[i-1],deg=0)
            phi2=np.angle(S21[i],deg=0)
            delta = np.abs(phi2-phi1)
            if delta>0.5*np.pi and delta<1.5*np.pi:
                S21[i:]=-S21[i:]
        block=spfile(freqs=freqs, portsayisi=2)
        block.sdata[:,0]=S11
        block.sdata[:,3]=S22
        block.sdata[:,1]=S21
        block.sdata[:,2]=S21
        return block
    else:
        return (S11,S22,S21)

def thru_line_deembedding(thru_filename, line_filename):
    """Extraction of transition s-parameters from THRU and LINE measurements. Transitions on both sides are assumed to be identical. For output *spfile* objects, port-1 is launcher side and port-2 is transmission line side. The length difference between LINE and THRU should be ideally :math:`\lambda/4`.
    The reference impedance for the 2. port of the transition should be the same as the characteristic impedance of the interconnecting line. So the reference impedances of the output *spfile* should be adjusted (without renormalizing s-parameters) after calling this function. The minimum frequency in the S-Parameter files should be such that the phase difference between the measurements should be smaller than 2:math:`\pi`.

    Args:
        thru_filename (str): 2-Port S-Parameter filename of THRU measurement
        line_filename (str): 2-Port S-Parameter filename of LINE measurement

    Returns:
        tuple(spfile, numpy.ndarray): 2-Element tuple of (transition spfile, complex phase vector (:math:`-\gamma l`) of connecting line of LINE standard (in radian))
    """

    if isinstance(thru_filename, spfile):
        Tthru = thru_filename
    else:
        Tthru = spfile(thru_filename)
    if isinstance(thru_filename, spfile):
        Tline = line_filename
    else:
        Tline = spfile(line_filename)
    output=deepcopy(Tthru)
    freqs = Tthru.get_frequency_list()
    Tthru.set_frequency_points(freqs)
    Tline.set_frequency_points(freqs)

    def kokbul(a,b,c):
        # Used to find -1/S22
        r1=(-b+np.sqrt(b*b-4*a*c))/2/a
        r2=(-b-np.sqrt(b*b-4*a*c))/2/a
        koks=[]
        for i in range(len(r1)):
            if np.abs(r1[i])<np.abs(r2[i]):
                koks.append(r2[i])
            else:
                koks.append(r1[i])
        return np.array(koks)

    TT = Tthru.s2t()
    TL = Tline.s2t()

    A = Tthru.T(1,1)
    B = Tthru.T(1,2)
    D = Tthru.T(2,2)

    A1 = Tline.T(1,1)
    B1 = Tline.T(1,2)
    D1 = Tline.T(2,2)

    x1 = B*D1-B1*D
    x2 = 2*B*x1+D*(A*D1-A1*D)
    x3 = (B*B+A*D)*x1

    b = kokbul(x1,x2,x3)
    a=(b+B)/D
    alpha=b*D/((b+B)*D1-B1*D)
    gammaL = np.log(alpha)
    gammaLr = np.real(gammaL)
    gammaLi = np.imag(gammaL)
    gammaL = gammaLr+1j*np.unwrap(gammaLi,discont=np.pi)
    t21 = np.sqrt(D/(b**2-1))
    t11 = t21*a
    t22 = t21*b
    t12 = (t11*t22-1)/t21
    s11, s12, s21, s22 = tuple(network.t2s_list([t11,t12,t21,t22]))

    for i in range(len(freqs)):
        output.sdata[i,:]=[s11[i],s12[i],s21[i],s22[i]]

    return output, gammaL

def trl_launcher_extraction(thru_filename, line_filename, reflect_filename, refstd=False):
    """Extraction of launcher s-parameters by THRU, LINE, REFLECT calibration. For output *spfile* objects, port-1 is launcher side and port-2 is transmission line side.
    Reference: TRL algorithm to de-embed a RF test fixture.pdf

    Args:
        thru_filename (str): 2-Port S-Parameter filename of THRU measurement
        line_filename (str): 2-Port S-Parameter filename of LINE measurement
        reflect_filename (str): 2-Port S-Parameter filename of REFLECT measurement
        refstd (boolean): True if OPEN is used as REFLECT standard and False (default) if SHORT is used

    Returns:
        tuple(spfile, sofile, numpy.ndarray): 3-Element tuple of (left side launcher spfile, right side launcher spfile, phase vector of connecting line of LINE standard (in radian) )
    """
    Tthru = spfile(thru_filename)
    Tline = spfile(line_filename)
    Tref = spfile(reflect_filename)
    freqs = Tthru.get_frequency_list()
    Tline.set_frequency_points(freqs)
    Tref.set_frequency_points(freqs)
    Tlauncherout=deepcopy(Tthru)
    Tlauncherin=deepcopy(Tthru)

    Tm = (-Tthru+Tline).s2t()
    Tn = (Tline-Tthru).s2t()

    def kokbul(a,b,c):
        r1=(-b+np.sqrt(b*b-4*a*c))/2/a
        r2=(-b-np.sqrt(b*b-4*a*c))/2/a
        if np.abs(r1)<np.abs(r2):
            return r1,r2
        else:
            return r2,r1

    s11=Tthru.S(1,1)
    s12=Tthru.S(1,2)
    s21=Tthru.S(2,1)
    s22=Tthru.S(2,2)

    s11s=Tref.S(1,1)
    s22s=Tref.S(1,1)

    faz=[]

    for i in range(len(freqs)):
        # m11,m12,m21,m22 = tuple(Tm.tdata[i])
        m22,m21,m12,m11 = tuple(Tm.tdata[i])
        x1,x2=kokbul(m21,m11-m22,-m12) # T12/T11=x1, T22/T21=x2
        # n11,n12,n21,n22 = tuple(Tn.tdata[i])
        n22,n21,n12,n11 = tuple(Tn.tdata[i])
        x3,x4=kokbul(n21,n11-n22,-n12) # T12_/T11_=x3, T22_/T21_=x4
        x5 = (1+x3*s11[i])/s21[i] # T11/T11_=x5
        x6 = s12[i]/(s22[i]+x2) # T21/T22_=x6

        A=x5*(s22s[i]+x1)/(s22s[i]+x2)
        B=x6*x4*(1+x4*s11s[i])/(1+x3*s11s[i])
        x7 = np.sqrt(A/B) # T21_/T11_=x7

        gref=x7*(1+x4*s11s[i])/(1+x3*s11s[i])
        if refstd==False:
            if np.real(gref)>0:
                x7=-x7
        else:
            if np.real(gref)<0:
                x7=-x7
        K=np.sqrt(1/x7/(x4-x3)) # choose sign of K depending on phase at DC
        # Tlauncher.sdata[i,:] = network.t2s_list([K*x5,K*x1*x5,K*x4*x6*x7,K*x2*x4*x6*x7])
        Tlauncherout.sdata[i,:] = network.t2s_list([K*x2*x4*x6*x7,K*x4*x6*x7,K*x1*x5,K*x5])
        K=1/K/x7/(x4-x3)
        Tlauncherin.sdata[i,:] = network.t2s_list([K,-K*x7,-K*x3,K*x4*x7])
        faz.append(np.log(m12/x1+m22)) #gamma.L
        # faz.append(-np.log(m11+x2*m21)) #gamma.L

    if np.abs(Tlauncherout.S(2,1,"phase"))[0]>90:
        Tlauncherout.sdata[:,1]=-Tlauncherout.sdata[:,1]
        Tlauncherout.sdata[:,2]=-Tlauncherout.sdata[:,2]
    if np.abs(Tlauncherin.S(2,1,"phase"))[0]>90:
        Tlauncherin.sdata[:,1]=-Tlauncherin.sdata[:,1]
        Tlauncherin.sdata[:,2]=-Tlauncherin.sdata[:,2]

    Tlauncherout.snp2smp([2,1],1)
    return Tlauncherout,Tlauncherin,np.array(faz)

class spfile:
    """ Class to process Touchstone files.

    \todo
    TODO:

    """
    def __init__(self,dosya="",freqs=None,portsayisi=1,satiratla=0):
        self.dataformat="DB"
        self.freq_unit="HZ"
        self.refimpedance=[]
        self.file_name=""
        self.sdata=[]
        self.ydata = None
        self.zdata = None
        self.abcddata=[]
        self.tdata=[]
        self.portnames=[]
        self.inplace=1
        self.YZ_OK=0
        self.ABCD_OK=0
        self.sym_smatrix = None
        self.sym_parameters = dict()
        self.undefinedYindices=set()
        self.undefinedZindices=set()
        self.formulation=1  # 1: "power-wave"
                            # 2: "pseudo-wave"
                            # 3: "HFSS pseudo-wave"
        self.comments=[] # comments in the file before the format line
        self.sparam_gen_func = None # this function generates smatrix of the network given the frequency
        self.sparam_mod_func = None # this function modifies smatrix of the network given the frequency
        self.params = {}
        self.dosya=dosya
        if not dosya=="":
            self.dosyaoku(dosya,satiratla)
        else:
            self.refimpedance=[50.0]*portsayisi
            self.FrequencyPoints=np.array(freqs)
            self.port_sayisi=portsayisi
            ns = len(self.FrequencyPoints)
            self.normalized=1 # normalized to 50 ohm if 1
            if ns>0:
                self.sdata=np.zeros((ns,portsayisi**2),complex)
            for i in range(portsayisi):
                self.portnames=[""]*portsayisi

    def copy(self):
        return deepcopy(self)

    def set_inplace(self, inplace):
        self.inplace = inplace

    def set_formulation(self, formulation):
        self.formulation = formulation

    def get_formulation(self):
        return self.formulation

    def change_formulation(self, formulation):
        refimpedance = deepcopy(self.refimpedance)
        self.change_ref_impedance(50.0)
        self.formulation = formulation
        self.change_ref_impedance(refimpedance)

    def copy_data_from_spfile(self,local_i,local_j,source_i,source_j,sourcespfile):
        """ This method copies S-Parameter data from another SPFILE object
        """
        local_column=(local_i-1)*self.port_sayisi+(local_j-1)
        self.sdata[:,local_column]=sourcespfile.data_array(dataformat="COMPLEX",i=source_i,j=source_j)

    def set_frequencies_wo_recalc(self,freqs):
        """Directly sets the frequencies of this network, but does not re-calculate s-parameters.

        Args:
            freqs (list or ndarray): New frequency values
        """
        self.FrequencyPoints=deepcopy(freqs)

    def getfilename(self):
        return self.file_name

    def getdataformat(self):
        return self.dataformat

    def setdataformat(self,dataformat):
        self.dataformat=dataformat

    def setdatapoint(self,m, indices,x):
        """Set the value for some part of S-Parameter data.

            .. math:: S_{i j}[m:m+len(x)]=x

        Args:
            m (int): Starting frequency indice
            indices (tuple of int): Parameters to be set (i,j)
            x (number or list): New value. If this is a number, it is converted to a list.
        """
        if isintance(x,(int, float,complex)):
            x=[x]
        (i,j) = indices
        for k in range(len(x)):
            self.sdata[k+m,(i-1)*self.port_sayisi+(j-1)]=x[k]

    def column_of_data(self,i,j):
        """Gets the indice of column at *sdata* matrix corresponding to :math:`S_{i j}`
        For internal use of the library.

        Args:
            i (int): First index
            j (int): Second index

        Returns:
            int: Index of column
        """
        return (i-1)*self.port_sayisi+(j-1)

    def get_sym_smatrix(self):
        return self.sym_smatrix

    def set_sym_smatrix(self,SM):
        """This function is used to set arithmetic expression for S-Matrix, if S-Matrix is defined using symbolic variables.

        Args:
            SM (sympy.Matrix): Symbolic ``sympy.Matrix`` expression for S-Parameter matrix
        """
        self.sym_smatrix = SM
        f = sp.Symbol("f")
        if f in self.sym_smatrix.free_symbols:
            self.sparam_gen_func = lambda x : self.sym_smatrix.subs((f,x)).evalf()
        else:
            self.sparam_gen_func = lambda x : self.sym_smatrix.evalf()
        self.set_frequency_points(self.get_frequency_list())

    def get_sym_parameters(self):
        """This function is used to get the values of symbolic variables of the network.

        Returns:
            dict: This is a dictionary containing the values of symbolic variables of the network
        """
        return self.sym_parameters

    def set_sym_parameters(self,paramdict):
        """This function is used to set the values of symbolic variables of the network. This is used if the S-Matrix of the network is defined by an arithmetic expression containing symbolic variables. This property is used in conjunction with *sympy* library for symbolic manipulation. Arithmetic expression for S-Matrix is defined by ``set_sym_smatrix`` function.

        Args:
            paramdict (dict): This is a dictionary containing the values of symbolic variables of the network
        """
        self.sym_parameters = paramdict
        if self.sym_smatrix:
            sym_smatrix = self.sym_smatrix
            if len(self.sym_parameters.keys())>0:
                sym_smatrix.subs(list(self.sym_parameters.items()))
            f = sp.Symbol("f")
            if f in self.sym_smatrix.free_symbols:
                self.sparam_gen_func = lambda x : self.sym_smatrix.subs((f,x)).evalf()
            else:
                self.sparam_gen_func = lambda x : self.sym_smatrix.evalf()
            self.set_frequency_points(self.get_frequency_list())

    def set_sparam_gen_func(self,func = None):
        """This function is used to set the function that generates s-parameters from frequency.

        Args:
            func (function, optional): function to be set. Defaults to None.
        """
        self.sparam_gen_func = func

    def set_sparam_mod_func(self,func = None):
        """This function is used to set the function that generates s-parameters from frequency.

        Args:
            func (function, optional): function to be set. Defaults to None.
        """
        self.sparam_mod_func = func

    def set_smatrix_at_frequency_point(self,indices,smatrix):
        """Set S-Matrix at frequency indices

        Args:
            indices (list): List of frequency indices
            smatrix (numpy.matrix): New S-Matrix value which is to be set at all *indices*
        """
        c=np.shape(smatrix)[0]
        smatrix=smatrix.reshape(c*c)
        if isinstance(indices,int):
            indices=[indices]
        for i in indices:
            self.sdata[i,:]=np.array(smatrix)

    def snp2smp(self,ports,inplace=-1):
        """This method changes the port numbering of the network
        port j of new network corresponds to ports[j] in old network.

        if the length of "ports" argument is lower than number of ports, remaining ports are terminated with current reference impedances and number of ports are reduced.

        Args:
            ports (list): New port order
            inplace (int, optional): Object editing mode. Defaults to -1.

        Returns:
            spfile: Modified spfile object
        """
        if inplace==-1: inplace=self.inplace
        if inplace==0:  obj = deepcopy(self); obj.set_inplace(1)
        else:           obj = self
        ns = len(obj.FrequencyPoints)
        ps=obj.port_sayisi
        newps=len(ports)
        sdata=obj.sdata # to speed-up
        new_sdata=np.zeros([ns,newps*newps]).astype(complex)
        for i in range(newps):
            for j in range(newps):
                n=(i)*newps+(j)
                m=(ports[i]-1)*ps+(ports[j]-1)
                new_sdata[:,n]=sdata[:,m]
        obj.sdata=new_sdata
        obj.port_sayisi=newps
        obj.refimpedance=[obj.refimpedance[x-1] for x in ports]
        names=obj.portnames
        obj.portnames=[names[ports[i]-1] for i in range(obj.port_sayisi)]
        obj.YZ_OK=0
        obj.ABCD_OK=0
        return obj

    def scaledata(self,scale=1.0, dataindices=None):
        if not dataindices:
            for i in range(self.port_sayisi**2):
                self.sdata[:,i]=self.sdata[:,i]*scale

    def get_no_of_ports(self):
        return self.port_sayisi

    def dosyayi_tekrar_oku(self):
        self.dosyaoku(self.file_name)

    def dosyaoku(self,file_name,satiratla=0):

        self.file_name=file_name
        ext=file_name.split(".")[-1]
        if ext[-1].lower()=="p":
            try:
                ps=self.port_sayisi=int("".join(ext[1:-1]))
            except:
                print("Error determining port number\n")
                return
        else: # for port numbers between 10-99 for which the extension is .s23 for example
            try:
                ps=self.port_sayisi=int("".join(ext[1:]))
            except:
                print("Error determining port number\n")
                return
        try:
            f=open(file_name,'r')
        except:
            print("Error opening the file: "+file_name+"\n")
            return  0
        linesread=f.readlines()[satiratla:]
        lsonuc=[]
        lines=[]
        lfrekans=[]
        portnames = defaultdict(str)
        pat=r"\!\s+Port\[(\d+)\]\s+=\s+(.+):?(\d+)?" # Matching pattern for port names (for files exported from HFSS)
        pat1=r"\!\s+(\w+)\s+=\s+(-?\d*\.?\w+)\s*" # Matching pattern for parameters (for files exported from HFSS)
        imps=[]
        gammas=[]
        index=0
        while index < len(linesread):
            x = linesread[index].strip()
            matchObj = re.match(pat,x)
            matchObj1 = re.match(pat1,x)
            if matchObj:
                if len(matchObj.groups())==4:
                    # group(3) is the mode number for DrivenModal designs
                    portnames[ int(matchObj.group(1)) ] = matchObj.group(2)+"_"+matchObj.group(3)
                else:
                    portnames[ int(matchObj.group(1)) ] = matchObj.group(2)
            elif matchObj1:
                self.params[ matchObj1.group(1) ] = matchObj1.group(2)
            elif len(x)>0:
                if x[0]!="!":
                    lines.append(x.split("!")[0].strip())
                else:
                    if "Gamma" in x:
                        tempportgamma = x.replace("!","").replace("Gamma","").strip().split()
                        while index<len(linesread)-1:
                            index=index+1
                            x = linesread[index].strip()
                            if len(x)==0:
                                continue
                            if x[0]!="!":
                                index=index-1
                                break
                            else:
                                if "Port" in x:
                                    break
                                else:
                                    tempportgamma = tempportgamma + x.replace("!","").strip().split()

                        tempportgamma=[float(tempportgamma[i])+1j*float(tempportgamma[i+1]) for i in range(round(len(tempportgamma)/2))]

                        #  The reason of the following correction: N-port Gamma and Impedances comments are exported in the Driven Terminal mode as NxN while in Modal mode there are exported as 1xN. (https://github.com/scikit-rf/scikit-rf/issues/354)
                        if len(tempportgamma)==ps*ps:
                            tempportgamma=[tempportgamma[i*i-1] for i in range(1,ps+1)]
                        gammas.append(tempportgamma)
                    if "Port Impedance" in x:
                        tempportimp = x.replace("!","").replace("Port Impedance","").strip().split()
                        while index<len(linesread)-1:
                            index=index+1
                            x = linesread[index].strip()
                            if len(x)==0:
                                continue
                            if x[0]!="!":
                                index=index-1
                                break
                            else:
                                tempportimp = tempportimp + x.replace("!","").strip().split()
                        tempportimp=[float(tempportimp[2*i])+1j*float(tempportimp[2*i+1]) for i in range(round(len(tempportimp)/2))]

                        #  The reason of the following correction: N-port Gamma and Impedances comments are exported in the Driven Terminal mode as NxN while in Modal mode there are exported as 1xN. (https://github.com/scikit-rf/scikit-rf/issues/354)
                        if len(tempportimp)==ps*ps:
                            tempportimp=[tempportimp[i*i-1] for i in range(1,ps+1)]
                        imps.append(tempportimp)
            index=index+1
        if len(imps)>0:
            imps2=[[arr[i] for arr in imps] for i in range(ps)]
        if len(gammas)>0:
            self.gammas=[[arr[i] for arr in gammas] for i in range(ps)]
        else:
            self.gammas=[]
        self.portnames=[""]*ps
        for i,pn in portnames.items():
            self.portnames[i-1]=pn
        x=lines[0]
        self.dataformat,self.freq_unit,refimpedance=parseformat(x)
        if refimpedance==None:
            if len(imps)==0:
                self.refimpedance=[50.0]*ps
            else:
                self.refimpedance = imps2
        else:
            self.refimpedance=[refimpedance]*ps
        lines.remove(x)
        datalar=np.array((" ".join(lines)).split(),dtype=float)
        k=(2*ps**2+1)
        b=len(datalar)//k # nokta sayisi
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

        lfrekans=datalar[:b,0]*fcoef[self.freq_unit]

        if self.dataformat=="RI":
            lsonuc=[datalar[:b,2*i+1]+datalar[:b,2*i+2]*1.0j for i in range(ps**2)]
        elif self.dataformat=="DB":
            lsonuc=[10**((datalar[:b,2*i+1]/20.0))*np.cos(datalar[:b,2*i+2]*np.pi/180)+10**((datalar[:b,2*i+1]/20.0))*np.sin(datalar[:b,2*i+2]*np.pi/180)*1j for i in range(ps**2)]
        else:
            lsonuc=[datalar[:b,2*i+1]*np.cos(datalar[:b,2*i+2]*np.pi/180)+datalar[:b,2*i+1]*np.sin(datalar[:b,2*i+2]*np.pi/180)*1.0j for i in range(ps**2)]
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
        return 1

    def get_undefinedYindices(self):
        return self.undefinedYindices

    def get_undefinedZindices(self):
        return self.undefinedZindices

    def Ffunc(self,imp):
        """Calculates F-matrix in a, b definition of S-Parameters. For internal use of the library.

                .. math::
                    a=F(V+Z_rI)

                    b=F(V-Z_r^*I)
        Args:
            imp (ndarray): Zref, Reference impedance array for which includes the reference impedance for each port.

        Returns:
            numpy.matrix: F-Matrix
        """
        if self.formulation == 1:
            F=np.matrix(np.diag(np.sqrt((0.25/abs(imp.real)))))
        elif self.formulation == 2:
            F=np.matrix(np.diag(np.sqrt(abs(imp.real))/2/abs(imp)))
        elif self.formulation == 3:
            F=np.matrix(np.diag(np.sqrt(0.25/imp)))
        return F

    def calc_syz(self,input="S",indices=None):
        """This function calculates 2 of S, Y and Z parameters by the remaining parameter given.
        Y and Z-matrices calculated separately instead of calculating one and taking inverse. Because one of them may be undefined for some circuits.

        Args:
            input (str, optional): Input parameter type (should be S, Y or Z). Defaults to "S".
            indices (list, optional): If given, output matrices are calculated only at the indices given by this list. If it is None, then output matrices are calculated at all frequencies. Defaults to None.
        """
        if input=="Z" and self.zdata is None:
            print("Z matrix is not calculated before")
            return
        if input=="Y" and self.ydata is None:
            print("Y matrix is not calculated before")
            return
        imp=self.prepare_ref_impedance_array(self.refimpedance)
        impT=imp.T
        ps=self.port_sayisi
        ns=len(self.FrequencyPoints)
        if indices==None:
            indices=list(range(ns))
        birim=np.matrix(np.eye(ps))
        G=np.matrix(np.zeros((ps,ps),dtype=complex))
        F=np.matrix(np.zeros((ps,ps),dtype=complex))
        if input=="S":
            self.undefinedYindices.clear()
            self.undefinedZindices.clear()
        if input=="S":
            # self.zdata=np.ones((ns,ps**2),dtype=complex)
            # self.ydata=np.ones((ns,ps**2),dtype=complex)
            if self.zdata is None:
                self.zdata=np.ones((ns,ps**2),dtype=complex)
            if self.ydata is None:
                self.ydata=np.ones((ns,ps**2),dtype=complex)
            if self.zdata.shape != (ns,ps**2):
                self.zdata=np.ones((ns,ps**2),dtype=complex)
            if self.ydata.shape != (ns,ps**2):
                self.ydata=np.ones((ns,ps**2),dtype=complex)
            sdata=self.sdata
            for i in indices:
                G=np.matrix(np.diag(impT[:][i]))
                F=self.Ffunc(impT[:][i])
                smatrix = np.matrix(sdata[i,:]).reshape(ps,ps)
                try:
                    if self.formulation == 1:
                        ymatrix=F.I*(smatrix*G+G.conj()).I*(birim-smatrix)*F
                    else:
                        ymatrix=F.I*(smatrix*G+G  ).I*(birim-smatrix)*F
                    self.ydata[i,:]=ymatrix.reshape(ps**2)
                except:
                    print("Y-Matrix is undefined at frequency: {: f}\n".format(self.FrequencyPoints[i]))
                    self.undefinedYindices.add(i)
                    break

                try:
                    if self.formulation == 1:
                        zmatrix=F.I*(birim-smatrix).I*(smatrix*G+G.conj())*F
                        # zmatrix=(G+G.conj()).I*F.I*(G*smatrix+G.conj())*(birim-smatrix).I*F*(G+G.conj()) # gives the same result
                    else:
                        zmatrix=F.I*(birim-smatrix).I*(smatrix*G+G)*F
                    self.zdata[i,:]=zmatrix.reshape(ps**2)
                except:
                    print("Z-Matrix is undefined at frequency: {: f}\n".format(self.FrequencyPoints[i]))
                    self.undefinedZindices.add(i)
                    break

        elif input=="Z":
            zdata=self.zdata
            if self.ydata.shape != (ns,ps**2):
                self.ydata=np.ones((ns,ps**2),dtype=complex)
            if self.sdata.shape != (ns,ps**2):
                self.sdata=np.ones((ns,ps**2),dtype=complex)
            for i in indices:
                G=np.matrix(np.diag(impT[:][i]))
                F=self.Ffunc(impT[:][i])
                zmatrix=np.matrix(zdata[i,:]).reshape(ps,ps)
                try:
                    ymatrix=zmatrix.I
                    self.ydata[i,:]=ymatrix.reshape(ps**2)
                except:
                    print("Y-Matrix is undefined at frequency: {: f}\n".format(self.FrequencyPoints[i]))
                    self.undefinedYindices.add(i)
                if self.formulation == 1:
                    smatrix=F*(zmatrix-G.conj())*(zmatrix+G).I*F.I
                else:
                    smatrix=F*(zmatrix-G)*(zmatrix+G).I*F.I
                self.sdata[i,:]=smatrix.reshape(ps**2)

        elif input=="Y":
            ydata=self.ydata
            if self.zdata.shape != (ns,ps**2):
                zdata=np.ones((ns,ps**2),dtype=complex)
            if self.sdata.shape != (ns,ps**2):
                sdata=np.ones((ns,ps**2),dtype=complex)
            for i in indices:
                G=np.matrix(np.diag(impT[:][i]))
                F=self.Ffunc(impT[:][i])
                ymatrix=np.matrix(ydata[i,:]).reshape(ps,ps)
                try:
                    zmatrix=ymatrix.I
                    self.zdata[i,:]=zmatrix.reshape(ps**2)
                except:
                    print("Z-Matrix is undefined at frequency: {: f}\n".format(self.FrequencyPoints[i]))
                    self.undefinedZindices.add(i)
                if self.formulation == 1:
                    smatrix = F*(birim-G.H*ymatrix)*(birim+G*ymatrix).I*F.I
                else:
                    smatrix = F*(birim-G*ymatrix)*(birim+G*ymatrix).I*F.I
                self.sdata[i,:]=smatrix.reshape(ps**2)

    def calc_t_eigs(self,port1=1,port2=2):
        """ Eigenfunctions and Eigenvector of T-Matrix is calculated.
        Only power-wave formulation is implemented """
        self.s2abcd(port1,port2)
        for i in range(len(self.FrequencyPoints)):
            abcd=self.abcddata[i].reshape(2,2)
            T=network.abcd2t(abcd,[50.0+0j,50.0+0j])
            eigs,eigv=eig(T)

    def s2t(self):
        """Take inverse of 2-port data for de-embedding purposes.

        Args:
            inplace (int, optional): Object editing mode. Defaults to -1.

        Returns:
            spfile: Inverted 2-port spfile
        """
        ns=len(self.FrequencyPoints)
        self.tdata=np.zeros((ns,self.port_sayisi**2),complex)
        for i in range(ns):
            smatrix=np.matrix(self.sdata[i,:]).reshape(2,2)
            sm = network.s2t(smatrix)
            self.tdata[i,:] = sm.reshape(4)
        return self

    def inverse_2port(self,inplace=-1):
        """Take inverse of 2-port data for de-embedding purposes.

        Args:
            inplace (int, optional): Object editing mode. Defaults to -1.

        Returns:
            spfile: Inverted 2-port spfile
        """
        if inplace==-1: inplace=self.inplace
        if inplace==0:  obj = deepcopy(self); obj.set_inplace(1)
        else:           obj = self
        obj.change_ref_impedance(50.0)
        ns=len(obj.FrequencyPoints)
        for i in range(ns):
            smatrix=np.matrix(obj.sdata[i,:]).reshape(2,2)
            # sm = network.t2s(network.s2t(smatrix).I)
            sm = network.abcd2s(network.s2abcd(smatrix).I)
            obj.sdata[i,:] = sm.reshape(4)
        return obj

    def s2abcd(self,port1=1,port2=2):
        """S-Matrix to ABCD matrix conversion between 2 chosen ports. Other ports are terminated with reference impedances

        Args:
            port1 (int, optional): Index of Port-1. Defaults to 1.
            port2 (int, optional): Index of Port-2. Defaults to 2.

        Returns:
            numpy.matrix: ABCD data. Numpy.matrix of size (ns,4) (ns: number of frequencies). Each row contains (A,B,C,D) numbers in order.
        """
        ns=len(self.FrequencyPoints)
        abcddata=np.ones((ns,4),dtype=complex)
        tempsp=self.snp2smp([port1, port2],inplace=0)
        tempsp.change_ref_impedance([50.0,50.0],1)
        birim=np.matrix(np.eye(2))
        for i in range(ns):
            smatrix=np.matrix(tempsp.sdata[i,:]).reshape(2,2)
            ABCD = network.s2abcd(smatrix)
            abcddata[i,:]=ABCD.reshape(4)
        self.abcddata = abcddata
        return abcddata

    def s2abcd2(self,port1=1,port2=2):
        """S-Matrix to ABCD matrix conversion between 2 chosen ports. Other ports are terminated with reference impedances

        Args:
            port1 (int, optional): Index of Port-1. Defaults to 1.
            port2 (int, optional): Index of Port-2. Defaults to 2.

        Returns:
            numpy.matrix: ABCD data. Numpy.matrix of size (ns,4) (ns: number of frequencies). Each row contains (A,B,C,D) numbers in order.
        """
        ns=len(self.FrequencyPoints)
        imp=self.prepare_ref_impedance_array(self.refimpedance)
        imp=np.concatenate(([imp[port1-1][:]],[imp[port2-1][:]]),axis=0)
        impT=imp.T
        abcddata=np.ones((ns,4),dtype=complex)
        sdata=np.ones((ns,4),dtype=complex)
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
                ABCD = network.y2abcd(ymatrix)
            else:
                if self.formulation == 1:
                    zmatrix=F.I*(birim-smatrix).I*(smatrix*G+G.conj())*F
                else:
                    zmatrix=F.I*(birim-smatrix).I*(smatrix*G+G)*F
                ABCD = network.z2abcd(zmatrix)
            abcddata[i,:]=ABCD.reshape(4)
        self.abcddata = abcddata
        return abcddata

    def input_impedance(self,k):
        """Input impedance at port k. All ports are terminated with reference impedances.

        Args:
            port (int): Port number for input impedance.

        Returns:
            numpy.ndarray: Array of impedance values for all frequencies
        """
        imp=self.prepare_ref_impedance_array(self.refimpedance)
        Zr = imp[k-1]
        T=self.S(k,k)
        if self.formulation==1:
            Z=(Zr.conj()+Zr*T)/(1-T)
        else:
            Z=Zr*(1+T)/(1-T)
        return Z

    def load_impedance(self,Gamma_in,port1=1,port2=2):
        """Calculates termination impedance at port2 that gives Gamma_in reflection coefficient at port1.

        Args:
            Gamma_in (float,ndarray): Required reflection coefficient.
            port1 (int): Source port.
            port2 (int): Load port.

        Returns:
            numpy.ndarray: Array of reflection coeeficient of termination at port2
        """
        s11=self.S(port1,port1)
        s22=self.S(port2,port2)
        s12=self.S(port1,port2)
        s21=self.S(port2,port1)
        Gamma_Load=(Gamma_in-s11)/(s12*s21+s22*(Gamma_in-s11))
        imp=self.prepare_ref_impedance_array(self.refimpedance)
        Zr = imp[port2-1]
        if self.formulation==1:
            Z=(Zr.conj()+Zr*Gamma_Load)/(1-Gamma_Load)
        else:
            Z=Zr*(1+Gamma_Load)/(1-Gamma_Load)
        return Z

    def gmax(self,port1=1,port2=2, dB=True):
        """Calculates Gmax from port1 to port2. Other ports are terminated with current reference impedances. If dB=True, output is in dB, otherwise it is a power ratio.

        Args:
            port1 (int, optional): Index of input port. Defaults to 1.
            port2 (int, optional): Index of output port. Defaults to 2.
            dB (bool, optional): Enable dB output. Defaults to True.

        Returns:
            numpy.ndarray: Array of Gmax values for all frequencies
        """
        self.s2abcd(port1,port2)
        ns=len(self.FrequencyPoints)
        gain=[]
        for i in range(ns):
            ABCD = self.abcddata[i,:].reshape(2,2)
            St=network.abcd2s(ABCD,50.0).reshape(4)
            s11, s12, s21, s22 = tuple(flatten(St.tolist()))
            D=s11*s22-s12*s21
            K=(1-np.abs(s11)**2-np.abs(s22)**2+np.abs(D)**2)/np.abs(2*s12*s21)
            g=np.abs(s21)/np.abs(s12)*(K-csqrt(K*K-1))
            gain=gain+[g]
        gain = np.array(gain)
        if dB==True:
            return 10*np.log10(gain)
        else:
            return gain

    def get_port_names(self):
        """Get list of port names.
        """
        return self.portnames

    def set_port_name(self, name, i):
        """Set name of a specific port.

        Args:
            name(str): New name of the port
            i(int): Port number
        """
        self.portnames[i+1]=name

    def set_port_names(self, names):
        """Set port names with a list.

        Args:
            names(list): List of new names of the ports
        """
        self.portnames=names[:]

    def get_port_number_from_name(self,isim):
        """Index of first port index with name *isim*

        Args:
            isim (bool): Name of the port

        Returns:
            int: Port index if port is found, 0 otherwise
        """
        if self.portnames.count(isim)>0:
            return self.portnames.index(isim)+1
        else:
            return 0

    def gav(self,port1=1,port2=2, ZS=[], dB=True):
        """Available gain from port1 to port2. If dB=True, output is in dB, otherwise it is a power ratio.

            .. math:: G_{av}=\\frac{P_{av,toLoad}}{P_{av,fromSource}}

        Args:
            port1 (int, optional): Index of input port. Defaults to 1.
            port2 (int, optional): Index of output port. Defaults to 2.
            ZS (list or numpy.ndarray, optional): Impedance of input port. Defaults to current reference impedance.
            dB (bool, optional): Enable dB output. Defaults to True.

        Returns:
            numpy.ndarray: Array of Gmax values for all frequencies
        """
        imp=self.prepare_ref_impedance_array(self.refimpedance)
        if ZS==[]:
            ZS = imp[port1-1]
        ZS=np.array(ZS)
        tsp=self.change_ref_impedance(50.0,0).snp2smp([port1,port2],0)
        GS=(ZS-50.0)/(ZS+50.0)
        s11=tsp.S(1,1)
        s12=tsp.S(1,2)
        s21=tsp.S(2,1)
        s22=tsp.S(2,2)
        Gout=s22+s12*s21*GS/(1-s11*GS)
        gain=(1-np.abs(GS)**2)/np.abs(1-s11*GS)**2*np.abs(s21)**2/(1-np.abs(Gout)**2)
        if dB==True:
            return 10*np.log10(gain)
        else:
            return gain

    def gop(self,port1=1,port2=2, ZL=50.0, dB=True):
        """Operating power gain from port1 to port2 with load impedance of ZL. If dB=True, output is in dB, otherwise it is a power ratio.

            .. math:: G_{op}=\\frac{P_{toLoad}}{P_{toNetwork}}

        Args:
            port1 (int, optional): Index of input port. Defaults to 1.
            port2 (int, optional): Index of output port. Defaults to 2.
            ZL (ndarray or float, optional): Load impedance. Defaults to 50.0.
            dB (bool, optional): Enable dB output. Defaults to True.

        Returns:
            numpy.ndarray: Array of Gop values for all frequencies
        """
        tsp=self.snp2smp([port1,port2],0)
        tsp.change_ref_impedance(50.0,1)
        s11=tsp.S(1,1)
        s12=tsp.S(1,2)
        s21=tsp.S(2,1)
        s22=tsp.S(2,2)
        GammaL=(ZL-50)/(ZL+50)
        Gammain=s11+s12*s21*GammaL/(1-s22*GammaL)
        gain=1/(1-np.abs(Gammain)**2)*np.abs(s21)**2*(1-np.abs(GammaL)**2)/np.abs(1-s22*GammaL)**2

        if dB==True:
            return 10*np.log10(gain)
        else:
            return gain

    def gop2(self,port1=1,port2=2, ZL=50.0, dB=True):
        """Operating power gain from port1 to port2 with load impedance of ZL. If dB=True, output is in dB, otherwise it is a power ratio.

            .. math:: G_{op}=\\frac{P_{toLoad}}{P_{toNetwork}}

        Args:
            port1 (int, optional): Index of input port. Defaults to 1.
            port2 (int, optional): Index of output port. Defaults to 2.
            ZL (ndarray or float, optional): Load impedance. Defaults to 50.0.
            dB (bool, optional): Enable dB output. Defaults to True.

        Returns:
            numpy.ndarray: Array of Gop values for all frequencies
        """
        tsp=self.snp2smp([port1,port2],0)
        tsp.change_ref_impedance(50.0,1)
        s11=tsp.S(1,1)
        s12=tsp.S(1,2)
        s21=tsp.S(2,1)
        s22=tsp.S(2,2)
        z11=tsp.Z(1,1)
        z12=tsp.Z(1,2)
        z21=tsp.Z(2,1)
        z22=tsp.Z(2,2)
        RL=np.real(ZL)
        XL=np.imag(ZL)
        GammaL=(ZL-50)/(ZL+50)
        Gammain=s11+s12*s21*GammaL/(1-s22*GammaL)
        Zin=50*(1+Gammain)/(1-Gammain)
        Rin=np.real(Zin)
        gain=RL/Rin*np.abs(z21/(z22+ZL))**2

        if dB==True:
            return 10*np.log10(gain)
        else:
            return gain


    def conj_match_uncoupled(self,ports=[],inplace=-1, noofiters=50):
        """Sets the reference impedance for given ports as the complex conjugate of output impedance at each port.
        The ports are assumed to be uncoupled. Coupling is taken care of by doing the same operation multiple times.

        Args:
            ports (list,optional): [description]. Defaults to all ports.
            inplace (int, optional): Object editing mode. Defaults to -1.
            noofiters (int, optional): Numberof iterations. Defaults to 50.

        Returns:
            spfile object with new s-parameters
        """
        if inplace==-1: inplace=self.inplace
        if inplace==0:  obj = deepcopy(self); obj.set_inplace(1)
        else:           obj = self
        if ports==[]:
            ports=list(range(1,self.port_sayisi+1))
        for _ in range(noofiters):
            imp = list(obj.refimpedance)
            for p in ports:
                imp[p-1]=np.conj(obj.input_impedance(p))
            obj.change_ref_impedance(imp)
        return obj

    def Z_conjmatch(self,port1=1,port2=2):
        """Calculates source and load reflection coefficients for simultaneous conjugate match.

        Args:
            port1 (int, optional): [description]. Defaults to 1.
            port2 (int, optional): [description]. Defaults to 2.

        Returns:
            2-tuple of numpy.arrays (GS, GL):
                - GS: Reflection coefficient at Port-1
                - GL: Reflection coefficient at Port-2
        """
        s11,s12,s21,s22 = self.return_s2p(port1,port2)
        D=s11*s22-s12*s21
        c1=s11-D*s22.conj()
        c2=s22-D*s11.conj()
        b1=1+np.abs(s11)**2-np.abs(s22)**2-np.abs(D)**2
        b2=1+np.abs(s22)**2-np.abs(s11)**2-np.abs(D)**2
        GS=b1/2/c1
        bool1=np.array([2*(cc>0)-1 for cc in b1])
        GS=GS-np.sqrt(b1**2-4*np.abs(c1)**2+0j)/2/c1*bool1
        GL=b2/2/c2
        bool2=np.array([2*(cc>0)-1 for cc in b2])
        GL=GL-np.sqrt(b2**2-4*np.abs(c2)**2+0j)/2/c2*bool2
        return (GS,GL)

    def gt(self,port1=1,port2=2, ZS=[], ZL=[], dB=True):
        """This method calculates transducer gain (GT) from port1 to port2. Source and load impedances can be specified independently. If any one of them is not specified, current reference impedance is used for that port. Other ports are terminated by reference impedances. This calculation can also be done using impedance renormalization.

            .. math:: G_{av}=\\frac{P_{load}}{P_{av,fromSource}}

        Args:
            port1 (int, optional): Index of source port. Defaults to 1.
            port2 (int, optional): Index of load port. Defaults to 2.
            dB (bool, optional): Enable dB output. Defaults to True.
            ZS (float, optional): Source impedance. Defaults to 50.0.
            ZL (float, optional): Load impedance. Defaults to 50.0.

        Returns:
            numpy.ndarray: Array of GT values for all frequencies
        """
        imp=self.prepare_ref_impedance_array(self.refimpedance)
        if ZS==[]:
            ZS = imp[port1-1]
        if ZL==[]:
            ZL = imp[port2-1]
        tsp=self.change_ref_impedance(50.0,0).snp2smp([port1,port2],0)
        GS=(ZS-50.0)/(ZS+50.0)
        GL=(ZL-50.0)/(ZL+50.0)
        s11=tsp.S(1,1)
        s12=tsp.S(1,2)
        s21=tsp.S(2,1)
        s22=tsp.S(2,2)
        Gout=s22+s12*s21*GS/(1-s11*GS)
        gain=(1-np.abs(GS)**2)/np.abs(1-s11*GS)**2*np.abs(s21)**2*(1-np.abs(GL)**2)/np.abs(1-Gout*GL)**2
        if dB==True:
            return 10*np.log10(gain)
        else:
            return gain

    def interpolate_data(self, datain, freqs):
        """Calculate new data corresponding to new frequency points *freqs* by interpolation from original data corresponding to current frequency points of the network.

        Args:
            data (numpy.ndarray or list): Original data specified at current frequency points of the network.
            freqs (numpy.ndarray or list): New frequency list.

        Returns:
            numpy.ndarray: New data corresponding to *freqs*
        """
        data=np.array(datain)
        if "scipy.interpolate" in sys.modules:
            tck_db = scipy.interpolate.splrep(self.FrequencyPoints,np.real(data),s=0,k=1)  # s=0, smoothing off, k=1, order of spline
            newdatar = scipy.interpolate.splev(freqs,tck_db,der=0)  # the order of derivative of spline
            tck_db = scipy.interpolate.splrep(self.FrequencyPoints,np.imag(data),s=0,k=1)  # s=0, smoothing off, k=1, order of spline
            newdatai = scipy.interpolate.splev(freqs,tck_db,der=0)  # the order of derivative of spline
            return newdatar+1j*newdatai
        else:
            return np.interp(freqs, self.FrequencyPoints, np.real(data))+1j*np.interp(freqs, self.FrequencyPoints, np.imag(data))

    def return_s2p(self,port1=1,port2=2):
        i21=(port1-1)*self.port_sayisi+(port2-1)
        i12=(port2-1)*self.port_sayisi+(port1-1)
        i11=(port1-1)*self.port_sayisi+(port1-1)
        i22=(port2-1)*self.port_sayisi+(port2-1)
        sdata=self.sdata
        s11=sdata[:,i11]
        s21=sdata[:,i21]
        s12=sdata[:,i12]
        s22=sdata[:,i22]
        return s11,s12,s21,s22

    def stability_factor_mu1(self,port1=1,port2=2):
        """Calculates :math:`\mu_1` stability factor, from port1 to port2. Other ports are terminated with reference impedances.

        Args:
            port1 (int, optional): Index of source port. Defaults to 1.
            port2 (int, optional): Index of load port. Defaults to 2.

        Returns:
            numpy.ndarray: Array of stability factor for all frequencies
        """
        s11,s12,s21,s22 = self.return_s2p(port1,port2)
        d=s11*s22-s12*s21
        mu1=(1.0-abs(s11)**2)/(abs(s22-d*s11.conjugate())+abs(s21*s12))
        return mu1

    def stability_factor_mu2(self,port1=1,port2=2):
        """Calculates :math:`\mu_2` stability factor, from port1 to port2. Other ports are terminated with reference impedances.

        Args:
            port1 (int, optional): Index of source port. Defaults to 1.
            port2 (int, optional): Index of load port. Defaults to 2.

        Returns:
            numpy.ndarray: Array of stability factor for all frequencies
        """
        s11,s12,s21,s22 = self.return_s2p(port1,port2)
        d=s11*s22-s12*s21
        mu2=(1.0-abs(s22)**2)/(abs(s11-d*s22.conj())+abs(s21*s12))
        return mu2

    def stability_factor_k(self,port1=1,port2=2):
        """Calculates *k* stability factor, from port1 to port2. Other ports are terminated with reference impedances.

        Args:
            port1 (int, optional): Index of source port. Defaults to 1.
            port2 (int, optional): Index of load port. Defaults to 2.

        Returns:
            numpy.ndarray: Array of stability factor for all frequencies
        """
        s11,s12,s21,s22 = self.return_s2p(port1,port2)
        d=s11*s22-s12*s21
        K=((1-abs(s11)**2-abs(s22)**2+abs(d)**2)/(2*abs(s21*s12)))
        return K

    def change_ref_impedance(self,Znew,inplace=-1):
        """Changes reference impedance and re-calculates S-Parameters.

        Args:
            Znew (float or list): New Reference Impedance. Its type can be:
                - float: In this case Znew value is used for all ports
                - list: In this case each element of this list is assgined to different ports in order as reference impedance. Length of *Znew* should be equal to number of ports. If an element of the list is None, then the reference impedance for corresponding port is not changed.

        Returns:
            spfile: The spfile object with new reference impedance
        """
        if inplace==-1: inplace=self.inplace
        if inplace==0:  obj = deepcopy(self); obj.set_inplace(1)
        else:           obj = self
        if isinstance(Znew, (list,np.ndarray)):
            for i in range(len(Znew)):
                if Znew[i] is None:
                    Znew[i]=obj.refimpedance[i]
        imp=obj.prepare_ref_impedance_array(obj.refimpedance)
        impT=imp.T
        impnew=obj.prepare_ref_impedance_array(Znew)
        impnewT=impnew.T
        ps = obj.port_sayisi
        birim=np.matrix(np.eye(ps))
        for i in range(len(obj.FrequencyPoints)):
            if obj.formulation == 1:
                G=np.matrix( np.diag( (impnewT[:][i]-impT[:][i])/(impnewT[:][i]+impT[:][i].conj()) ) )
                F=obj.Ffunc(impT[:][i])
                Fnew=obj.Ffunc(impnewT[:][i])
                A = Fnew.I*F*(birim-G.conj())
                S = np.matrix(obj.sdata[i,:]).reshape(ps,ps)
                C1 = (S-G.conj())
                C2 = (birim-G*S).I
                Snew = A.I*C1*C2*A.conj()
            else: # TODO: Should be derived, tried and tested
                G=np.matrix( np.diag( (impnewT[:][i]-impT[:][i])/(impnewT[:][i]+impT[:][i]) ) )
                F=obj.Ffunc(impT[:][i])
                Fnew=obj.Ffunc(impnewT[:][i])
                A = Fnew.I*F*(birim-G)
                S = np.matrix(obj.sdata[i,:]).reshape(ps,ps)
                C1 = (S-G)
                C2 = (birim-G*S).I
                Snew = A.I*C1*C2*A
            obj.sdata[i,:]=Snew.reshape(ps**2)
        if isinstance(Znew,(complex, float, int)):
            obj.refimpedance=np.ones(ps,dtype=complex)*Znew
        else:
            obj.refimpedance = deepcopy(Znew)
        return obj

    def prepare_ref_impedance_array(self,imparray):
        """Turns reference impedance array which is composed of numbers,arrays, functions or 1-ports to numerical array which
        is composed of numbers and arrays. It is made sure that :math:`Re(Z)\neq 0`. Mainly for internal use.

        Args:
            imparray (list): List of impedance array

        Returns:
            numpy.ndarray: Calculated impedance array
        """
        newarray=[]
        if isinstance(imparray,(float,complex,int)):
            imparray = [imparray]*self.port_sayisi
        for i in range(self.port_sayisi):
            if isinstance(imparray[i],spfile):
                # imparray[i].calc_syz("S")
                # newarray.append(np.array([x+(x.real==0)*1e-8 for x in imparray[i].data_array(dataformat="COMPLEX",syz="Z",i=1,j=1, frekanslar=self.FrequencyPoints) ]))
                imparray[i].change_ref_impedance(50.0)
                Gamma = imparray[i].data_array(dataformat="COMPLEX",M="S",i=1,j=1, frekanslar=self.FrequencyPoints)
                Zin = 50.0*(1.0+Gamma)/(1.0-Gamma)
                newarray.append(np.array([x+(x.real==0)*1e-8 for x in Zin ]))
            elif inspect.isfunction(imparray[i]):
                newarray.append(np.array([x+(x.real==0)*1e-8 for x in imparray[i](self.FrequencyPoints)]))
            elif isinstance(imparray[i],(float,complex,int)):
                newarray.append(np.ones(len(self.FrequencyPoints))*(imparray[i]+(imparray[i].real==0)*1e-8))
            elif isinstance(imparray[i],(list, np.ndarray)):
                newarray.append(np.array(imparray[i]))
        return np.array(newarray)

    def ImpulseResponse(self,i=2,j=1,dcinterp=1,dcvalue=0.0,MaxTimeStep=1.0,FreqResCoef=1.0, Window="blackman"):
        """Calculates impulse response of :math:`S_{i j}`

        Args:
            i (int, optional): Port-1. Defaults to 2.
            j (int, optional): Port-2. Defaults to 1.
            dcinterp (int, optional): If 1, add DC point to interpolation. Defaults to 1.
            dcvalue (float, optional): dcvalue to be used at interpolation if *dcinterp=0*. Defaults to 0.0. This value is appended to :math:`S_{i j}` and the rest is left to interpolation in *data_array* function.
            MaxTimeStep (float, optional): Not used for now. Defaults to 1.0.
            FreqResCoef (float, optional): Coeeficient to increase the frequency resolution by interpolation. Defaults to 1.0 (no interpolation).
            Window (str, optional): Windows function to prevent ringing. Defaults to "blackman". Other windows will be added later.

        Returns:
            9-tuple: The elements of the tuple are the following in order:
                1. Raw frequency data used as input
                2. Window array
                3. Time array
                4. Time-Domain Waveform of Impulse Response
                5. Time-Domain Waveform of Impulse Input
                6. Time step
                7. Frequency step
                8. Size of input array
                9. Max Value of Impulse Input
        """

        nn=int(FreqResCoef*self.FrequencyPoints[-1]/(self.FrequencyPoints[-1]-self.FrequencyPoints[0])*len(self.FrequencyPoints)) #data en az kac noktada temsil edilecek
        fmax = self.FrequencyPoints[-1]
        # Frequency step
        df=(self.FrequencyPoints[-1]/(nn))

        # Generate nn frequency points starting from df/2 and ending at fmax with df spacing
        nfdata=np.linspace((df),self.FrequencyPoints[-1],nn)
        # Get complex data in frequency domain for positive frequencies
        rawdata=self.data_array(dataformat="COMPLEX",M="S",i=i,j=j, frekanslar=nfdata,DCInt=dcinterp,DCValue=dcvalue)

        # Handle negative frequencies, Re(-w)=Re(w),Im(-w)=-Im(w), and prepare data array for ifft
        # Zero padding on frequency data to obtain a smooth time-domain plot
        N=2**(int((np.log10(nn)/np.log10(2.0)))+10)
        data=np.zeros(N,dtype=complex)
        data[0] = dcvalue
        data[1:(nn+1)] = rawdata[:]
        data[(N-nn):(N)] = [x.conj() for x in rawdata[::-1]]
        #   Extrapolate to upper frequencies (Best to avoid)
        #   Windowing to mitigate sudden spectral data change at high frequency side
        # window= get_window(Window,2*nn+1)
        window= blackman_window(2*nn+1)
        # window=scipy.signal.get_window("blackman",2*nn+1)
        datawindow=np.zeros(N,dtype=complex)
        datawindow[1:(nn+1)] = window[nn+1:]
        datawindow[N-nn:] = window[:nn]
        #   IFFT
        F_data=np.fft.ifft(data*datawindow)
        #   Normalization with impulse response by calculating amplitude of impulse
        #   response assuming the frequency coefficients of 1.
        data=np.zeros(N,dtype=complex)
        data[:(nn+1)] = 1.0
        data[N-nn:] = 1.0
        Vintime=np.fft.ifft(data*datawindow)
        Norm = np.max(np.real(Vintime[0]))
        Vintime = Vintime/Norm
        F_data = F_data/Norm
        #   Determine time step
        dt=1./N/df
        #   Generate Time Axis
        shift = int(5.0/fmax/dt)
        shift = 0
        # Vintime=np.concatenate((Vintime[-shift:],Vintime[:N-shift]))
        # F_data=np.concatenate((F_data[-shift:],F_data[:N-shift]))
        timeline=np.linspace(-shift*dt,dt*(N-1)-shift*dt,N)
        return (rawdata,window,timeline,F_data,Vintime,dt,df,N,Norm)

    def __sub__(self,SP2):
        """Implements SP1-SP2.
        Deembeds SP2 from port-2 of SP1.
        Port ordering is as follows:
        (1)-SP1-(2)---(1)-SP2-(2)
        SP1 is *self*.

        Args:
            SP2 (spfile): Deembedded spfile network

        Returns:
            spfile: The resulting of deembedding process
        """
        if (self.port_sayisi!=2 or SP2.port_sayisi!=2):
            print("Both networks should be two-port")
            return 0
        sonuc=deepcopy(self)
        refimp_port1 = sonuc.refimpedance[0]
        SP2.set_frequency_points(sonuc.FrequencyPoints)
        refimp_port2 = SP2.refimpedance[1]
        sonuc.change_ref_impedance(50.0)
        SP2.change_ref_impedance(50.0)
        sonuc.s2abcd()
        SP2.s2abcd()
        for i in range(len(sonuc.FrequencyPoints)):
            abcd1=np.matrix(sonuc.abcddata[i].reshape(2,2))
            abcd2=np.matrix(SP2.abcddata[i].reshape(2,2))
            abcd=abcd1*abcd2.I
            s=network.abcd2s(abcd,50.0)
            sonuc.abcddata[i]=abcd.reshape(4)
            sonuc.sdata[i]=s.reshape(4)
            sonuc.YZ_OK=0
        sonuc.change_ref_impedance([refimp_port1,refimp_port2])
        return sonuc

    def __neg__(self):
        """Calculates an spfile object for two-port networks which is the inverse of this network. This is used to use + and - signs to cascade or deembed 2-port blocks.

        Returns:
            spfile:
                1. *None* if number of ports is not 2.
                2. *spfile* which is the inverse of the spfile object operated on.
        """
        if (self.port_sayisi!=2):
            print("Network should be two-port")
            return None
        output = deepcopy(self)
        output.set_inplace(1)
        output.inverse_2port()
        return output

    def __add__(self,SP2):
        """Implements SP1+SP2.
        Cascades SP2 to port-2 of SP1.
        Port ordering is as follows:
        (1)-SP1-(2)---(1)-SP2-(2)
        SP1 is *self*.

        Args:
            SP2 (spfile): Appended spfile network

        Returns:
            spfile: The result of cascade of 2 networks
        """
        if (self.port_sayisi!=2 or SP2.port_sayisi!=2):
            print("Both networks should be two-port")
            return 0
        sonuc=deepcopy(self)
        sonuc.set_inplace(1)
        refimp_port1 = sonuc.refimpedance[0]
        if len(sonuc.FrequencyPoints)>len(SP2.FrequencyPoints):
            print("Uyari: ilk devrenin frekans nokta sayisi, 2. devreninkinden fazla")
        SP2.set_frequency_points(sonuc.FrequencyPoints)
        refimp_port2 = SP2.refimpedance[1]
        sonuc.change_ref_impedance(50.0)
        SP2.change_ref_impedance(50.0)
        sonuc.s2abcd()
        SP2.s2abcd()
        for i in range(len(sonuc.FrequencyPoints)):
            abcd1=np.matrix(sonuc.abcddata[i].reshape(2,2))
            abcd2=np.matrix(SP2.abcddata[i].reshape(2,2))
            abcd=abcd1*abcd2
            s=network.abcd2s(abcd,50.0)
            sonuc.abcddata[i]=abcd.reshape(4)
            sonuc.sdata[i]=s.reshape(4)
            sonuc.YZ_OK=0
        sonuc.change_ref_impedance([refimp_port1,refimp_port2])
        return sonuc

    def check_passivity(self):
        """This method determines the frequencies and frequency indices at which the network is not passive. Condition written in:
        Fast Passivity Enforcement of S-Parameter Macromodels by Pole Perturbation.pdf
        For a better discussion: "S-Parameter Quality Metrics (Yuriy Shlepnev)"

        Returns:
            3-tuple of lists: For non-passive frequencies (indices, frequencies, eigenvalues)
        """
        frekanslar=[] # frequency points at which the network is non-passive
        indices=[]    # frequency indices at which the network is non-passive
        eigenvalues=[]
        ps=self.port_sayisi
        for i in range(len(self.FrequencyPoints)):
            smatrix=np.matrix(self.sdata[i,:]).reshape(ps,ps)
            tempmatrix=smatrix.H*smatrix
            eigs,_=eig(tempmatrix)
            if np.max(np.abs(eigs)) > 1:
                frekanslar.append(self.FrequencyPoints[i])
                indices.append(i)
                eigenvalues.append(sort(eigs))
        return  indices,self.FrequencyPoints(indices),eigenvalues

    def restore_passivity(self, inplace=-1):
        """Make the network passive by minimum modification.
        Method reference:
        "Fast and Optimal Algorithms for Enforcing Reciprocity, Passivity and Causality in S-parameters.pdf"

        Args:
            inplace (int, optional): Object editing mode. Defaults to -1.

        Returns:
            spfile: Passive network object
        """
        if inplace==-1: inplace=self.inplace
        if inplace==0:  obj = deepcopy(self); obj.set_inplace(1)
        else:           obj = self
        frekanslar=[] # frequency points at which the network is non-passive
        indices=[]    # frequency indices at which the network is non-passive
        indices,frequencies,eigenvalues = self.check_passivity()
        ps=obj.port_sayisi
        for i in indices:
            smatrix=np.matrix(obj.sdata[i,:]).reshape(ps,ps)
            P, D, Q = np.linalg.svd(smatrix)
            dS = P @ np.diag([(np.abs(x)-1.0)*(np.abs(x)>0) for x in eigenvalues[i]]) @ Q
            smatrix=smatrix-dS
            obj.sdata[i,:]=smatrix.reshape(ps**2)
        return obj

    def restore_passivity2(self):
        """**Obsolete**
         Bu metod S-parametre datasinin pasif olmadigi frekanslarda
        S-parametre datasina mumkun olan en kucuk degisikligi yaparak
        S-parametre datasini pasif hale getirir.
        Referans:
        Restoration of Passivity In S-parameter Data of Microwave Measurements.pdf
        """
        _, indices=self.check_passivity()
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
        smatrix=np.matrix(np.eye(ps),dtype=complex)
        perturbation =np.zeros((ps,ps),dtype=complex)
        for index in indices:
            for y in range((ps)**2):
                p=sdata[index,y]
                # p1=p.real
                # p2=p.imag
                smatrix[ (y/ps) , y%ps]=p

            while (1):
                tempmatrix=np.matrix(np.eye(ps).astype(complex))-smatrix.H*smatrix
                eigs,eigv=eig(tempmatrix)
                eigsl,eigvl=eig(tempmatrix.T)
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

                def constraint1(x, grad=0):
                    """ Eigenvalue'nun yeni degerinin pozitif olmasi icin bu deger >0 olmali"""
                    return -coef+sum([x[i]*c[i] for i in range(2*t)])
                def constraint1_der(x):
                    """ constraint1 fonksiyonunun turevi """
                    return c
                def func_for_minimize(x,grad=0):
                    """ bu deger minimize edilmeli
                        yani S-matrixteki degisim gerekli minimum duzeyde olmali
                    """
                    return sum([y**2 for y in x])
                def func_for_minimize_der(x):
                    """ func_for_minimize fonksiyonunun turevi """
                    return 2*x
                cons=({ 'type'   :   'ineq',
                        'fun'    :   constraint1,
                        'jac'    :   constraint1_der },)
                if "scipy.optimize" in sys.modules:
                    from scipy.optimize import minimize
                    res = minimize(func_for_minimize, xvar, jac = func_for_minimize_der,constraints = cons,  method = 'SLSQP', options={'disp': False})
                    x = res.x
                else:
                    try:
                        import nlopt
                        opt = nlopt.opt(nlopt.GN_ESCH, len(2*t))
                        opt.add_inequality_constraint(constraint1)
                        # opt.set_maxeval(1000)
                        # opt.set_maxtime(5)
                        x = opt.optimize(xvar)
                    except:
                        print("Error at root finding with NLOPT")

                for y in range(t):
                    perturbation[(y/ps),y%ps]=x[2*y]+x[2*y+1]*1.0j
                smatrix=smatrix+perturbation
                #tempmatrix=np.matrix(np.eye(ps).astype(complex))-smatrix.H*smatrix
                #eigs,eigv=eig(tempmatrix)
                #print "eigs_after_iter ",eigs
            for y in range((ps)**2):
                sdata[index,y]=smatrix[ (y/ps) , y%ps]

    def write2file(self,newfilename="",parameter="S",freq_unit="",dataformat=""):
        """This function writes a parameter (S, Y or Z) file. If the filename given does not have the proper filename extension, it is corrected.

        Args:
            newfilename (str, optional): Filename to be written. Defaults to "".
            parameter (str, optional): Parameter to be written (S, Y or Z). Defaults to "S".
            freq_unit (str, optional): Frequency unit (GHz, MHz, kHz or Hz). Defaults to "Hz".
            dataformat (str, optional): Format of file DB, RI or MA. Defaults to "".
        """
        if newfilename=="":
            newfilename = self.file_name
        if freq_unit=="":
            freq_unit=self.freq_unit
        freq_unit = freq_unit.upper()
        if dataformat=="":
            dataformat=self.dataformat

        ext = "s"+str(self.port_sayisi)
        if len(ext)<3: ext=ext+"p"
        if newfilename[-4:].lower() != "."+ext:
            newfilename = newfilename+"."+ext

        f=open(newfilename,'w')
        f.write("# "+freq_unit+" "+parameter+" "+dataformat+" R "+str(self.refimpedance[0].real)+"\n")
        for i,portname in enumerate(self.portnames):
            f.write(f"! Port[{i+1}] = {portname}\n")
        frekanslar=self.FrequencyPoints
        if parameter!="S":
            self.calc_syz()
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
        temp=(1./fcoef[freq_unit])
        max_params_per_line=4
        if ps==3:
            max_params_per_line=3
        if (dataformat=="RI"):
            for x in range(len(frekanslar)):
                # print("\n"+f"{frekanslar[x]*temp:.2f}"+"    ", end='', file=f)
                print("\n%-12.3f"%(frekanslar[x]*temp)+"    ", end='', file=f)
                for j in range(ps**2):
                    print("%-12.12f    %-12.12f" % (np.real(data[x,j]),np.imag(data[x,j])), end='', file=f)
                    if ((j+1)%max_params_per_line==0 and j<(ps**2-1)):
                        print("\n", end='', file=f)
                    elif j!=ps**2-1:
                        print("  ", end='', file=f)
        elif (dataformat=="MA"):
            for x in range(len(frekanslar)):
                # print("\n"+f"{frekanslar[x]*temp:.2f}"+"    ", end='', file=f)
                print("\n%-12.3f"%(frekanslar[x]*temp)+"    ", end='', file=f)
                # print("\n"+str(frekanslar[x]*temp)+"    ", end='', file=f)
                for j in range(ps**2):
                    print("%-12.12f    %-12.12f" % (np.abs(data[x,j]),np.angle(data[x,j],deg=1)), end='', file=f)
                    if ((j+1)%max_params_per_line==0 and j<(ps**2-1)):
                        print("\n", end='', file=f)
                    elif j!=ps**2-1:
                        print("  ", end='', file=f)
        else:
            for x in range(len(frekanslar)):
                # print("\n"+f"{frekanslar[x]*temp:.2f}"+"    ", end='', file=f)
                print("\n%-12.3f"%(frekanslar[x]*temp)+"    ", end='', file=f)
                # print("\n"+str(frekanslar[x]*temp)+"    ", end='', file=f)
                for j in range(ps**2):
                    print("%-12.12f    %-12.12f" % (20*np.log10(np.abs(data[x,j])),np.angle(data[x,j],deg=1)), end='', file=f)
                    if ((j+1)%max_params_per_line==0 and j<(ps**2-1)):
                        print("\n     ", end='', file=f)
                    elif j<(ps**2-1):
                        print("  ", end='', file=f)
        print("", file=f)
        f.close()

    def get_frequency_list(self):
        """Returns the frequency list of network

        Returns:
            numpy.array: Frequency list of network
        """
        return self.FrequencyPoints

    def connect_2_ports_list(self,conns,inplace=-1):
        """Short circuit ports together one-to-one. Short circuited ports are removed.
        Ports that will be connected are given as tuples in list *conns*
        i.e. conns=[(p1,p2),(p3,p4),..]
        The order of remaining ports is kept.
        Reference: QUCS technical.pdf, S-parameters in CAE programs, p.29

        Args:
            conns (list of tuples): A list of 2-tuples of integers showing the ports connected
            inplace (int, optional): Object editing mode. Defaults to -1.

        Returns:
            spfile: New spfile object
        """
        if inplace==-1: inplace=self.inplace
        if inplace==0:  obj = deepcopy(self); obj.set_inplace(1)
        else:           obj = self
        for i in range(len(conns)):
            k,m = conns[i]
            obj.connect_2_ports(k,m)
            for j in range(i+1,len(conns)):
                conns[j][0]=conns[j][0]-(conns[j][0]>k)-(conns[j][0]>m)
                conns[j][1]=conns[j][1]-(conns[j][1]>k)-(conns[j][1]>m)
        return obj

    def connect_2_ports(self,k,m,inplace=-1):
        """Port-m is connected to port-k and both ports are removed.
        Reference: QUCS technical.pdf, S-parameters in CAE programs, p.29

        Args:
            k (int): First port index to be connected.
            m (int): Second port index to be connected.
            inplace (int, optional): Object editing mode. Defaults to -1.

        Returns:
            spfile: New spfile object
        """
        if inplace==-1: inplace=self.inplace
        if inplace==0:  obj = deepcopy(self); obj.set_inplace(1)
        else:           obj = self
        k,m=min(k,m),max(k,m)
        newrefimpedance = list(obj.refimpedance[:k-1])+list(obj.refimpedance[k:m-1])+list(obj.refimpedance[m:])
        portnames = obj.portnames[:k-1]+obj.portnames[k:m-1]+obj.portnames[m:]
        names=defaultdict(str)
        obj.change_ref_impedance(50.0)
        ns = len(obj.FrequencyPoints)
        ps=obj.port_sayisi
        sdata=np.ones((ns,(ps-2)**2),dtype=complex)
        S = obj.S
        for i in range(1,ps-1):
            ii=i+(i>=k)+(i>=(m-1))
            for j in range(1,ps-1):
                jj=j+(j>=k)+(j>=(m-1))
                index = (ps-2)*(i-1)+(j-1)
                temp = S(k,jj)*S(ii,m)*(1-S(m,k))+S(m,jj)*S(ii,k)*(1-S(k,m))+S(k,jj)*S(m,m)*S(ii,k)+S(m,jj)*S(k,k)*S(ii,m)
                temp = S(ii,jj) + temp/((1-S(m,k))*(1-S(k,m))-S(k,k)*S(m,m))
                sdata[:,index] = temp
        obj.port_sayisi = ps-2
        obj.sdata = sdata
        obj.refimpedance=[50.0]*obj.port_sayisi
        obj.change_ref_impedance(newrefimpedance)
        obj.portnames=portnames
        return obj

    def connect_2_ports_retain(self,k,m,inplace=-1):
        """Port-m is connected to port-k and both ports are removed. New port becomes the last port of the circuit.
        Reference: QUCS technical.pdf, S-parameters in CAE programs, p.29

        Args:
            k (int): First port index to be connected.
            m (int): Second port index to be connected.
            inplace (int, optional): Object editing mode. Defaults to -1.

        Returns:
            spfile: New *spfile* object
        """
        if inplace==-1: inplace=self.inplace
        if inplace==0:  obj = deepcopy(self); obj.set_inplace(1)
        else:           obj = self
        ideal3port = spfile(freqs=obj.get_frequency_list(),portsayisi=3)
        ideal3port.set_smatrix_at_frequency_point(range(len(ideal3port.get_frequency_list())),network.idealNport(3))
        ps = obj.port_sayisi
        k,m=min(k,m),max(k,m)
        obj.connect_network_1_conn(ideal3port,m,1)
        obj.connect_2_ports(k,ps)
        return obj

    def connect_network_1_conn_retain(self,EX,k,m,inplace=-1):
        """Port-m of EX circuit is connected to port-k of this circuit. This connection point will also be a port.
        Remaining ports of EX are added to the port list of this circuit in order.
        Reference: QUCS technical.pdf, S-parameters in CAE programs, p.29

        Args:
            EX (spfile): External network to be connected to this.
            k (int): Port number of self to be connected.
            m (int): Port number of EX to be connected.
            inplace (int, optional): Object editing mode. Defaults to -1.
            preserveportnumbers1 (bool, optional): if True, the number of the first added port will be k. Defaults to False.

        Returns:
            spfile: Connected network
        """
        if inplace==-1: inplace=self.inplace
        if inplace==0:  obj = deepcopy(self); obj.set_inplace(1)
        else:           obj = self
        ideal3port = spfile(freqs=obj.get_frequency_list(),portsayisi=3)
        ideal3port.set_smatrix_at_frequency_point(range(len(ideal3port.get_frequency_list())),network.idealNport(3))
        EX.connect_network_1_conn(ideal3port,m,1)
        psex = EX.get_no_of_ports()
        obj.connect_network_1_conn(EX,k,psex)
        return obj

    def connect_network_1_conn(self,EX,k,m,inplace=-1, preserveportnumbers1= False):
        """Port-m of EX circuit is connected to port-k of this circuit. Both of these ports will be removed.
        Remaining ports of EX are added to the port list of this circuit in order.
        Reference: QUCS technical.pdf, S-parameters in CAE programs, p.29

        Args:
            EX (spfile): External network to be connected to this.
            k (int): Port number of self to be connected.
            m (int): Port number of EX to be connected.
            inplace (int, optional): Object editing mode. Defaults to -1.
            preserveportnumbers1 (bool, optional): if True, the number of the first added port will be k. Defaults to False.

        Returns:
            spfile: Connected network
        """
        if inplace==-1: inplace=self.inplace
        if inplace==0:  obj = deepcopy(self); obj.set_inplace(1)
        else:           obj = self
        newrefimpedance=list(obj.refimpedance[:k-1])+list(obj.refimpedance[k:])+list(EX.refimpedance[:m-1])+list(EX.refimpedance[m:])
        portnames=obj.portnames[:k-1]+obj.portnames[k:]+EX.portnames[:m-1]+EX.portnames[m:]
        EX.change_ref_impedance(50.0)
        obj.change_ref_impedance(50.0)
        EX.set_frequency_points(obj.FrequencyPoints)
        ps1=obj.port_sayisi
        ps2=EX.port_sayisi
        ps=ps1+ps2-2
        sdata=np.ones((len(obj.FrequencyPoints),ps**2),dtype=complex)
        S = obj.S
        for i in range(1,ps1):
            ii=i+(i>(k-1))
            for j in range(1,ps1):
                jj=j+(j>(k-1))
                index = (i-1)*ps+(j-1)
                sdata[:,index] = S(ii,jj)+S(k,jj)*EX.S(m,m)*S(ii,k)/(1.0-S(k,k)*EX.S(m,m))
        for i in range(1,ps2):
            ii=i+(i>(m-1))
            for j in range(1,ps1):
                jj=j+(j>(k-1))
                index = (i+ps1-1-1)*ps+(j-1)
                sdata[:,index] = S(k,jj) * EX.S(ii,m) / (1.0 - S(k,k) * EX.S(m,m))
        for i in range(1,ps1):
            ii=i+(i>(k-1))
            for j in range(1,ps2):
                jj=j+(j>(m-1))
                index = (i-1)*ps+(j+ps1-1-1)
                sdata[:,index] = EX.S(m,jj) * S(ii,k) / (1.0 - EX.S(m,m) * S(k,k))
        for i in range(1,ps2):
            ii=i+(i>(m-1))
            for j in range(1,ps2):
                jj=j+ (j>(m-1))
                index = (i+ps1-1-1)*ps+(j+ps1-1-1)
                sdata[:,index] = EX.S(ii,jj)+EX.S(m,jj)*S(k,k)*EX.S(ii,m)/(1.0-EX.S(m,m)*S(k,k))
        obj.port_sayisi=ps
        obj.sdata = sdata
        obj.refimpedance=[50.0]*obj.port_sayisi
        obj.change_ref_impedance(newrefimpedance)
        obj.portnames=portnames
        if preserveportnumbers1:
            portorder=list(range(1,ps+1))
            portorder.insert(k-1,portorder.pop(ps1-1))
            obj.snp2smp(portorder)
        return obj

    def add_abs_noise(self,dbnoise=0.1,phasenoise=0.1,inplace=-1):
        """This method adds random amplitude and phase noise to the s-parameter data.
        Mean value for both noises are 0.

        Args:
            dbnoise (float, optional): Standard deviation of amplitude noise in dB. Defaults to 0.1.
            phasenoise (float, optional): Standard deviation of phase noise in degrees. Defaults to 0.1.
            inplace (int, optional): object editing mode. Defaults to -1.

        Returns:
            spfile: object with noisy data
        """
        if inplace==-1: inplace=self.inplace
        if inplace==0:  obj = deepcopy(self); obj.set_inplace(1)
        else:           obj = self
        n=obj.port_sayisi**2
        ynew=[]
        sdata=np.zeros((len(obj.FrequencyPoints),n),dtype=complex)
        for j in range(n):
            ydb=np.array([20*np.log10(abs(obj.sdata[k,j])) for k in range(len(obj.FrequencyPoints))])
            yphase=np.array([np.angle(obj.sdata[k,j],deg=1)  for k in range(len(obj.FrequencyPoints))]) # degree
            ynew_db = ydb+dbnoise*np.random.normal(len(ydb))
            ynew_ph = yphase+phasenoise*np.random.normal(size=len(yphase))
            ynew_mag=10**((ynew_db/20.0))
            ynew=ynew_mag*(np.cos(ynew_ph*np.pi/180)+1.0j*np.sin(ynew_ph*np.pi/180))
            sdata[:,j]=ynew
        obj.sdata=sdata
        obj.YZ_OK=0
        obj.ABCD_OK=0
        return obj

    def smoothing(self,smoothing_length=5,inplace=-1):
        """This method applies moving average smoothing to the s-parameter data

        Args:
            smoothing_length (int, optional): Number of points used for smoothing. Defaults to 5.
            inplace (int, optional): object editing mode. Defaults to -1.

        Returns:
            spfile: Network object with smooth data
        """
        if inplace==-1: inplace=self.inplace
        if inplace==0:  obj = deepcopy(self); obj.set_inplace(1)
        else:           obj = self
        n=obj.port_sayisi**2
        ynew=[]
        sdata=np.zeros((len(obj.FrequencyPoints),n),dtype=complex)
        for j in range(n):
            ydb=np.array([20*np.log10(abs(obj.sdata[k,j])) for k in range(len(obj.FrequencyPoints))])
            yphase=np.unwrap([np.angle(obj.sdata[k,j],deg=0)  for k in range(len(obj.FrequencyPoints))])*180.0/np.pi # degree
            ynew_db=smooth(ydb,window_len=smoothing_length,window='hanning')
            ynew_ph=smooth(yphase,window_len=smoothing_length,window='hanning')
            ynew_mag=10**((ynew_db/20.0))
            ynew=ynew_mag*(np.cos(ynew_ph*np.pi/180)+1.0j*np.sin(ynew_ph*np.pi/180))
            sdata[:,j]=ynew
        obj.sdata=sdata
        obj.ABCD_OK=0
        obj.YZ_OK=0
        return obj

    def data_array(self,dataformat="DB",M="S",i=1,j=1, frekanslar=None,ref=None, DCInt=0,DCValue=0.0,smoothing=0, InterpolationConstant=0):
        """Return a network parameter between ports *i* and *j* (:math:`M_{i j}`) at specified frequencies in specified format.

        Args:
            dataformat (str, optional): Defaults to "DB". The format of the data returned. Possible values (case insensitive):
                -   "K": Stability factor of 2-port
                -   "MU1": Input stability factor of 2-port
                -   "MU2": Output stability factor of 2-port
                -   "VSWR": VSWR ar port i
                -   "MAG": Magnitude of :math:`M_{i j}`
                -   "DB": Magnitude of :math:`M_{i j}` in dB
                -   "REAL": Real part of :math:`M_{i j}`
                -   "IMAG": Imaginary part of :math:`M_{i j}`
                -   "PHASE": Phase of :math:`M_{i j}` in degrees between 0-360
                -   "UNWRAPPEDPHASE": Unwrapped Phase of :math:`M_{i j}` in degrees
                -   "GROUPDELAY": Group Delay of :math:`M_{i j}` in degrees
            M (str, optional): Defaults to "S". Possible values (case insensitive):
                -   "S": Return S-parameter data
                -   "Y": Return Y-parameter data
                -   "Z": Return Z-parameter data
                -   "ABCD": Return ABCD-parameter data
            i (int, optional): First port number. Defaults to 1.
            j (int, optional): Second port number. Defaults to 1. Ignored for *dataformat*="VSWR"
            frekanslar (list, optional): Defaults to []. List of frequencies in Hz. If an empty list is given, networks whole frequency range is used.
            ref (spfile, optional): Defaults to None. If given the data of this network is subtracted from the same data of *ref* object.
            DCInt (int, optional): Defaults to 0. If 1, DC point given by *DCValue* is used at frequency interpolation if *frekanslar* is not [].
            DCValue (complex, optional): Defaults to 0.0. DCValue that can be used for interpolation over frequency.
            smoothing (int, optional): Defaults to 0. if this is higher than 0, it is used as the number of points for smoothing.
            InterpolationConstant (int, optional): Defaults to 0. If this is higher than 0, it is taken as the number of frequencies that will be added between 2 consecutive frequency points. By this way, number of frequencies is increased by interpolation.

        Returns:
            numpy.array: Network data array
        """
        if i>self.port_sayisi or j>self.port_sayisi:
            print("Error: port index is higher than number of ports!"+"\t"+str(i)+"\t"+str(j))
            return []
        FORMAT=str.upper(dataformat)
        if FORMAT=="K":
            return self.stability_factor_k(frekanslar,i,j)
        if FORMAT=="MU1":
            return self.stability_factor_mu1(frekanslar,i,j)
        if FORMAT=="MU2":
            return self.stability_factor_mu2(frekanslar,i,j)
        if FORMAT=="VSWR" and i!=j:
            j=i
            return

        if frekanslar is None:
            frekanslar=self.FrequencyPoints
        if InterpolationConstant > 0:
            frekstep = frekanslar[1]-frekanslar[0]
            frekanslar = np.array(list(range((len(frekanslar)-1)*InterpolationConstant+1)))*frekstep/InterpolationConstant+frekanslar[0]
        x=self.FrequencyPoints
        lenx=len(x)
        dcdb=[]
        dcph=[]
        if DCInt==1:
            dcdb=[20*np.log10((np.abs(DCValue)+1e-8))]
            dcph=[np.angle(DCValue,deg=False)]
            x=np.append([0.0],x)
        n=(i-1)*self.port_sayisi+(j-1)
        ynew=[]
        mag_threshold=1.0e-10
        if (str.upper(M)=="Y" or str.upper(M)=="Z"):
            self.calc_syz("S")
        if str.upper(M)=="S" or FORMAT=="GROUPDELAY":
            ydb=dcdb+[20*np.log10(abs(self.sdata[k,n])+mag_threshold) for k in range(lenx)]
            yph=np.unwrap(dcph+[np.angle(self.sdata[k,n],deg=0)  for k in range(lenx)])*180.0/np.pi
        elif str.upper(M)=="Y":
            ydb=dcdb+[20*np.log10(abs(self.ydata[k,n])+mag_threshold) for k in range(lenx)]
            yph=np.unwrap(dcph+[np.angle(self.ydata[k,n],deg=0)  for k in range(lenx)])*180.0/np.pi
        elif str.upper(M)=="T":
            self.s2t()
            ydb=dcdb+[20*np.log10(abs(self.tdata[k,n])+mag_threshold) for k in range(lenx)]
            yph=np.unwrap(dcph+[np.angle(self.tdata[k,n],deg=0)  for k in range(lenx)])*180.0/np.pi
        elif str.upper(M)=="Z":
            ydb=dcdb+[20*np.log10(abs(self.zdata[k,n])+mag_threshold) for k in range(lenx)]
            yph=np.unwrap(dcph+[np.angle(self.zdata[k,n],deg=0)  for k in range(lenx)])*180.0/np.pi
        elif str.upper(M)=="ABCD":
            self.s2abcd()
            ydb=dcdb+[20*np.log10(abs(self.abcddata[k,n])+mag_threshold) for k in range(lenx)]
            yph=np.unwrap(dcph+[np.angle(self.abcddata[k,n],deg=0)  for k in range(lenx)])*180.0/np.pi

        if frekanslar is self.FrequencyPoints:
            ynew_db=array(ydb )
            ynew_ph=array(yph)
        elif len(self.FrequencyPoints)>1:
            # order = 2
            # tck_db = scipy.interpolate.InterpolatedUnivariateSpline(x,ydb,k=order)
            # ynew_db = tck_db(frekanslar)
            # tck_phase = scipy.interpolate.InterpolatedUnivariateSpline(x,yph,k=order)
            # ynew_ph = tck_phase(frekanslar)

            if "scipy" in sys.modules:
                tck_db = scipy.interpolate.CubicSpline(x,ydb,extrapolate=True)
                ynew_db = tck_db(frekanslar)
                tck_phase = scipy.interpolate.CubicSpline(x,yph,extrapolate=True)
                ynew_ph = tck_phase(frekanslar)
            else:
                ynew_db = np.interp(frekanslar, x, ydb)
                ynew_ph = np.interp(frekanslar, x, yph) #degrees
        else:
            ynew_db=array(ydb*len(frekanslar))
            ynew_ph=array(yph*len(frekanslar))

        if not ref==None:
            ynew_db=ynew_db-ref.data_array("DB",M,i,j,frekanslar)
            ynew_ph=ynew_ph-ref.data_array("UNWRAPPEDPHASE",M,i,j,frekanslar)

        if smoothing>0:
            if smoothing>lenx-1:
                smoothing=lenx-1
            ynew_db=smooth(ynew_db,window_len=smoothing,window='hanning')
            ynew_ph=smooth(ynew_ph,window_len=smoothing,window='hanning')
        if  FORMAT=="COMPLEX":
            if len(frekanslar)==0: # interpolasyon yok.
                ynew=self.sdata[:,n]
            else:
                ynew_mag=10**((ynew_db/20.0))
                ynew=ynew_mag*(np.cos(ynew_ph*np.pi/180.0)+1.0j*np.sin(ynew_ph*np.pi/180.0))
        elif FORMAT=="DB":
            ynew = ynew_db
        elif FORMAT=="MAG":
            ynew = 10**((ynew_db/20.0))
        elif FORMAT=="VSWR":
            mag = 10**((ynew_db/20.0))
            ynew=((1.0+mag)/(1.0-mag))
        elif FORMAT=="REAL":
            ynew1 = 10**((ynew_db/20.0))
            ynew=ynew1*np.cos(ynew_ph*np.pi/180.0)
        elif FORMAT=="IMAG":
            ynew1 = 10**((ynew_db/20.0))
            ynew=ynew1*np.sin(ynew_ph*np.pi/180.0)
        elif FORMAT=="PHASE":
            ynew = np.mod(ynew_ph,360.)
        elif FORMAT=="UNWRAPPEDPHASE":
            ynew = ynew_ph
        elif FORMAT=="GROUPDELAY":
            t=len(frekanslar)
            ynew=[1]*t
            for k in range(1,t-1):
                ynew[k]=-(ynew_ph[k+1]-ynew_ph[k-1])/(frekanslar[k+1]-frekanslar[k-1])/360.0
            ynew[0]=-(ynew_ph[1]-ynew_ph[0])/(frekanslar[1]-frekanslar[0])/360.0
            ynew[t-1]=-(ynew_ph[t-1]-ynew_ph[t-2])/(frekanslar[t-1]-frekanslar[t-2])/360.0
            ynew=np.array(ynew)
        return ynew

    def Extraction(self, measspfile):
        """Extract die S-Parameters using measurement data and simulated S-Parameters
        Port ordering in *measspfile* is assumed to be the same as this *spfile*.
        Remaining ports are ports of block to be extracted.
        See "Extracting multiport S-Parameters of chip" in technical document.

        Args:
            measspfile (spfile): *SPFILE* object of measured S-Parameters of first k ports

        Returns:
            spfile: *SPFILE* object of die's S-Parameters
        """
        refimpedance = self.refimpedance # save to restore later
        self.change_ref_impedance(50.0)
        measspfile.change_ref_impedance(50.0)
        measspfile.set_frequency_points(self.FrequencyPoints)
        k = measspfile.get_no_of_ports()
        ps = self.port_sayisi
        block = spfile(freqs = self.FrequencyPoints, portsayisi = ps-k)
        for i in range(len(self.FrequencyPoints)):
            ST = np.matrix(self.sdata[i,:]).reshape(ps,ps)
            S11 = ST[:k,:k]
            S12 = ST[:k,k:]
            S21 = ST[k:,:k]
            S22 = ST[k:,k:]
            SM = np.matrix(measspfile.sdata[i,:]).reshape(k,k)
            SC = (S21*(SM-S11).I*S12+S22).I
            block.set_smatrix_at_frequency_point(i,SC)
        self.change_ref_impedance(refimpedance)
        return block

    def UniformDeembed(self, phase, delay=False, deg=True,inplace=-1):
        """This function deembeds a phase length from all ports of S-Parameters.A positive phase means deembedding into the circuit.
        The Zc of de-embedding lines is the reference impedances of each port.

        Args:
            phase (float or list): Phase or delay to be deembedded.
                - If a number is given, it is used for all frequencies and ports
                - If a list is given, its size should be equal to the number of frequencies. If an element of list is number, it is used for all ports. If an element of the list is also a list, the elements size should be same as the number of ports.
            delay (bool, optional): If True, *phase* is assumed to be time delay, phase otherwise. Defaults to False.
            deg (bool, optional): if True, *phase* is assumed to be in radians if *delay* is False. Defaults to 0.
            inplace (int, optional): Object editing mode. Defaults to -1.

        Returns:
            spfile: De-embedded spfile
        """
        if inplace==-1: inplace=self.inplace
        if inplace==0:  obj = deepcopy(self); obj.set_inplace(1)
        else:           obj = self
        w = 2*np.pi*obj.FrequencyPoints
        N=len(obj.FrequencyPoints)
        ps = obj.port_sayisi
        coef = 1
        if deg:
            #phase = phase*np.pi/180.0
            coef = np.pi/180.0
        if isinstance(phase,(complex, float, int)):
            phase = N*[phase]
        elif isinstance(phase, list):
            if len(phase)==ps:
                phase = N*[phase]
            else:
                if not len(phase)==N:
                    print("Wrong size of phase parameter")
                    return
        for i in range(N):
            if isinstance(phase[i],(complex, float, int)):
                phase[i] = ps*[phase[i]]
            smatrix = np.matrix(obj.sdata[i,:]).reshape(ps,ps)
            if delay:
                PhaseMatrix = np.matrix(np.diag([np.exp(1j*w[i]*x) for x in phase[i]]))
            else:
                PhaseMatrix = np.matrix(np.diag([np.exp(1j*x*coef) for x in phase[i]]))
            Sm = PhaseMatrix*smatrix*PhaseMatrix
            obj.sdata[i,:]=Sm.reshape(ps**2)
        return obj

    def S(self,i=1,j=1,dataformat="COMPLEX",freqs=None):
        """Return :math:`S_{i j}` in format *dataformat*
        Uses *data_array* method internally. A convenience function for practical use.

        Args:
            i (int, optional): Port-1. Defaults to 1.
            j (int, optional): Port-2. Defaults to 1.
            dataformat (str, optional): See *dataformat* parameter of *data_array* method. Defaults to "COMPLEX".

        Returns:
            numpy.array: :math:`S_{i j}` as *dataformat*
        """
        return self.data_array(dataformat,"S",i,j,frekanslar=freqs)


    def T(self,i=1,j=1,dataformat="COMPLEX",freqs=None):
        """Return :math:`T_{i j}` in format *dataformat*
        Uses *data_array* method internally. A convenience function for practical use.

        Args:
            i (int, optional): Port-1. Defaults to 1.
            j (int, optional): Port-2. Defaults to 1.
            dataformat (str, optional): See *dataformat* parameter of *data_array* method. Defaults to "COMPLEX".

        Returns:
            numpy.array: :math:`T_{i j}` as *dataformat*
        """
        return self.data_array(dataformat,"T",i,j,frekanslar=freqs)

    def Z(self,i=1,j=1,dataformat="COMPLEX",freqs=None):
        """Return :math:`Z_{i j}` in format *dataformat*
        Uses *data_array* method internally. A convenience function for practical use.

        Args:
            i (int, optional): Port-1. Defaults to 1.
            j (int, optional): Port-2. Defaults to 1.
            dataformat (str, optional): See *dataformat* parameter of *data_array* method. Defaults to "COMPLEX".

        Returns:
            numpy.array: :math:`Z_{i j}` as *dataformat*
        """
        return self.data_array(dataformat,"Z",i,j,frekanslar=freqs)

    def Y(self,i=1,j=1,dataformat="COMPLEX",freqs=None):
        """Return :math:`Y_{i j}` in format *dataformat*
        Uses *data_array* method internally. A convenience function for practical use.

        Args:
            i (int, optional): Port-1. Defaults to 1.
            j (int, optional): Port-2. Defaults to 1.
            dataformat (str, optional): See *dataformat* parameter of *data_array* method. Defaults to "COMPLEX".

        Returns:
            numpy.array: :math:`Y_{i j}` as *dataformat*
        """
        return self.data_array(dataformat,"Y",i,j,frekanslar=freqs)

    def set_frequency_limits(self,flow,fhigh,inplace=-1):
        """Remove frequency points higher than *fhigh* and lower than *flow*.

        Args:
            flow (float): Lowest Frequency (Hz)
            fhigh (float): Highest Frequency (Hz)
            inplace (int, optional): Object editing mode. Defaults to -1.

        Returns:
            spfile: spfile object with new frequency points.
        """
        newfreqs=list(filter(lambda x:flow<=x<=fhigh,self.FrequencyPoints))
        return self.set_frequency_points(newfreqs,inplace)

    def set_frequency_points(self,frekanslar,inplace=-1):
        """Set new frequency points. if S-Parameter data generator function is available, use that to calculate new s-parameter data. If not, use interpolation/extrapolation.
        For new frequency points, S-Parameters and reference impedances which are in the form of array are re-calculated.

        Args:
            frekanslar (list): New frequency array
            inplace (int, optional): Object editing mode. Defaults to -1.

        Returns:
            spfile: spfile object with new frequency points.
        """
        if inplace==-1: inplace=self.inplace
        if inplace==0:  obj = deepcopy(self); obj.set_inplace(1)
        else:           obj = self
        if isinstance(self.refimpedance,(list,np.ndarray)):
            for i in range(obj.port_sayisi):
                if isinstance(self.refimpedance[i],(list,np.ndarray)):
                    self.refimpedance[i] = self.interpolate_data(self.refimpedance[i], frekanslar)
        if len(self.gammas)>0:
            for i in range(obj.port_sayisi):
                self.gammas[i] = self.interpolate_data(self.gammas[i], frekanslar)
        if obj.sparam_gen_func is not None:
            obj.FrequencyPoints=frekanslar
            ns =len(obj.FrequencyPoints)
            # self.nokta_sayisi = ns
            obj.sdata=np.zeros((ns,obj.port_sayisi**2),dtype=complex)
            for i in range(len(obj.FrequencyPoints)):
                obj.set_smatrix_at_frequency_point(i,obj.sparam_gen_func(obj.FrequencyPoints[i]))
        else:
            if len(obj.FrequencyPoints)==0:
                obj.FrequencyPoints=frekanslar
                ns = len(obj.FrequencyPoints)
                # self.nokta_sayisi = ns
                obj.sdata=np.zeros((ns,obj.port_sayisi**2),dtype=complex)
            else:
                sdata=np.zeros((len(frekanslar),obj.port_sayisi**2),dtype=complex)
                for i in range(1,obj.port_sayisi+1):
                    for j in range(1,obj.port_sayisi+1):
                        n=(i-1)*obj.port_sayisi+(j-1)
                        sdata[:,n]=obj.data_array("COMPLEX","S",i,j, frekanslar)
                obj.FrequencyPoints=frekanslar
                obj.sdata=sdata
                # self.nokta_sayisi=len(self.FrequencyPoints)
        return obj

    def set_frequency_points_array(self,fstart,fstop,NumberOfPoints,inplace=-1):
        """Set the frequencies of the object using start-end frequencies and number of points.

        Args:
            fstart ([type]): Start frequency.
            fstop ([type]): End frequency.
            NumberOfPoints (int): Number of frequencies.
            inplace (int, optional): Object editing mode. Defaults to -1.

        Returns:
            spfile: spfile object with new frequency points.
        """
        self.set_frequency_points(frekanslar=np.linspace(fstart,fstop,NumberOfPoints,endpoint=True),inplace=-1)
        return self

    def convert_s1p_to_s2p(self):
        #thru kalibrasyonla S21 olcumu  durumu icin, pasif devreler icin.
        self.port_sayisi=2
        temp=(self.port_sayisi)**2
        ns = len(self.FrequencyPoints)
        newdata=np.zeros([ns,temp]).astype(complex)
        newdata[:,0]=1e-6+0j
        newdata[:,1]=self.sdata[:,0]
        newdata[:,2]=self.sdata[:,0]
        newdata[:,3]=1e-6+0j
        self.sdata=newdata
        return self

# Factory Methods to practically create specialized objects
    @classmethod
    def microstripstep(cls, w1, w2, eps_r, h, t, freqs=None):
        """Create an ``spfile`` object corresponding to a microstrip step.

        Args:
            w1 (float): Width of microstrip line at port-1.
            w2 (float): Width of microstrip line at port-2.
            eps_r (float): Relative permittivity of microstrip substrate.
            h (float): Thickness of microstrip substrate.
            t (float): Thickness of metal.
            freqs (float, optional): Frequency list of object. Defaults to None. If not given, frequencies should be set later.

        Returns:
            spfile: An spfile object equivalent to microstrip step.
        """
        obj = cls(portsayisi=2)
        obj.set_sparam_gen_func(lambda x:network.abcd2s(tlines.microstrip_step_in_width(w1, w2, eps_r, h, t, x)))
        if freqs:
            obj.set_frequency_points(freqs)
        return obj

    @classmethod
    def striplinestep(cls, w1, w2, eps_r, h1, h2, t, freqs=None):
        """Create an ``spfile`` object corresponding to a stripline step

        Args:
            w1 (float): Width of stripline line at port-1.
            w2 (float): Width of stripline line at port-2.
            eps_r (float): Relative permittivity of stripline substrate.
            h (float): Thickness of stripline substrate.
            t (float): Thickness of metal.
            freqs (float, optional): Frequency list of object. Defaults to None. If not given, frequencies should be set later.

        Returns:
            spfile: An spfile object.
        """
        obj = cls(portsayisi=2)
        obj.set_sparam_gen_func(lambda x:network.abcd2s(tlines.stripline_step_in_width(w1, w2, eps_r, h1, h2, t, x)))
        if freqs:
            obj.set_frequency_points(freqs)
        return obj

    @classmethod
    def microstripline(cls, length, w, h, t, er, freqs=None):
        """Create an ``spfile`` object corresponding to a microstrip line.

        Args:
            length (float): Length of microstrip line.
            w (float): Width of microstrip line.
            h (float): Thickness of substrate.
            t (float): Thickness of metal.
            er (float): Relative permittivity of microstrip substrate.
            freqs (float, optional): Frequency list of object. Defaults to None. If not given, frequencies should be set later.

        Returns:
            spfile: An spfile object.
        """
        obj = cls(portsayisi=2)
        def spr(freq):
            Z, eeff = tlines.Z_eeff_disp_thick_microstrip(w, h, t, er, freq)
            theta=2*np.pi*freq*np.sqrt(eeff)/c0*length
            return network.abcd2s(network.tline(Z, theta))
        obj.set_sparam_gen_func(spr)
        if freqs:
            obj.set_frequency_points(freqs)
        return obj

    @classmethod
    def stripline(cls, length, w, er, h1, h2, t, freqs=None):
        """Create an ``spfile`` object corresponding to a stripline transmission line.

        Args:
            length (float): Length of cpwg line.
            w (float): Width of stripline.
            er (float): Relative permittivity of substrate.
            h1 (float): Thickness of substrate from bottom ground to bottom of line.
            h2 (float): Thickness of substrate from top line to top ground.
            t (float): Thickness of metal.
            freqs (float, optional): Frequency list of object. Defaults to None. If not given, frequencies should be set later.

        Returns:
            spfile: An spfile object.
        """
        obj = cls(portsayisi=2)
        def spr(freq):
            Z = tlines.Z_thick_offset_stripline(w, er, h1, h2, t)
            theta=2*np.pi*freq*np.sqrt(er)/c0*length
            return network.abcd2s(network.tline(Z, theta))
        obj.set_sparam_gen_func(spr)
        if freqs:
            obj.set_frequency_points(freqs)
        return obj

    @classmethod
    def cpwgline(cls, length, w, th, er, s, h, freqs=None):
        """Create an ``spfile`` object corresponding to a cpwg transmission line.

        Args:
            length (float): Length of cpwg line.
            w (float): Width of cpwg line.
            th (float): Thickness of metal.
            er (float): Relative permittivity of substrate.
            s (float): Gap of cpwg line.
            h (float): Thickness of substrate.
            freqs (float, optional): Frequency list of object. Defaults to None. If not given, frequencies should be set later.

        Returns:
            spfile: An spfile object.
        """
        obj = cls(portsayisi=2)
        def spr(freq):
            Z, eeff = tlines.Z_eeff_grounded_cpw_thick(w, th, er, s, h)
            theta=2*np.pi*freq*np.sqrt(eeff)/c0*length
            return network.abcd2s(network.tline(Z, theta))
        obj.set_sparam_gen_func(spr)
        if freqs:
            obj.set_frequency_points(freqs)
        return obj

if __name__ == '__main__':

    sptline=spfile(freqs=[10e9],portsayisi=2)
    theta=90
    # for i in range(len(frequencies)):
    sptline.set_smatrix_at_frequency_point(0,network.abcd2s(network.tline(50.0,60*np.pi/180.0),50.0))
    print(sptline.sdata)
    sptline.UniformDeembed(-90*np.pi/180.0)
    print(sptline.sdata)
    # sptline=spfile(noktasayisi=1,portsayisi=2)
    # sptline.set_smatrix_at_frequency_point(0,np.array([[0.2,0.1],[0.1,0.3]]))
    # sptline.set_frequency_points([60e9,70e9,80e9])
    # newf=np.array([60e9,70e9,80e9])
    # print(sptline.data_array("COMPLEX","S",2,1,newf))

    # sp = spfile(r"C:\\Users\\Erdoel\\Documents\\Works\\BGT61ATR24\\RX_total_7_R3_HFSSDesign4.s8p")

    # sp.connect_2_ports_retain(7,8)
    # sp.connect_2_ports(5,6)
    # import matplotqlib.pyplot as plt

    # plt.plot(sp.get_frequency_list(), sp.S(3,4,"DB"),label="s34")
    # plt.plot(sp.get_frequency_list(), sp.S(3,8,"DB"),label="s38")
    # plt.plot(sp.get_frequency_list(), sp.S(8,3,"DB"),label="s83")
    # plt.plot(sp.get_frequency_list(), sp.S(9,8,"DB"),label="s98")
    # plt.legend()
    # plt.grid()
    # plt.show()


    # sp = spfile(r"C:\\Users\\Erdoel\\Documents\\MATLAB\\Transition Design Tests\\QS2_ExtractedZ_chip.s1p")
    # import pysmith
    # ax = pysmith.get_smith(plt.figure(), 111)
    # s11=sp.S(1,1,"COMPLEX")
    # ax.plot(np.real(s11),np.imag(s11))
    # plt.show()
    # ideal3port_1 = spfile(noktasayisi=1,portsayisi=3)
    # ideal3port_1.set_smatrix_at_frequency_point(0,1/3*np.array([[-1,2,2],[2,-1,2],[2,2,-1]]))
    # ideal3port_2 = spfile(noktasayisi=1,portsayisi=3)
    # ideal3port_2.set_smatrix_at_frequency_point(0,1/3*np.array([[-1,2,2],[2,-1,2],[2,2,-1]]))
    # ideal3port_1.connect_network_1_conn(ideal3port_2,3,1)
    # print(ideal3port_1.S(1,1))
    # print(ideal3port_1.S(1,3))
    # print(ideal3port_1.S(3,1))
    # print(ideal3port_1.S(3,4))

    # ideal3port = spfile(noktasayisi=1,portsayisi=3)
    # ideal3port.set_smatrix_at_frequency_point(0,1/3*np.array([[-1,2,2],[2,-1,2],[2,2,-1]]))
    # ideal3port.set_frequency_points(sp.get_frequency_list() )
    # plt.plot(sp.get_frequency_list(), ideal3port.S(2,1,"DB"),label="s21")
    # plt.plot(sp.get_frequency_list(), ideal3port.S(1,1,"DB"),label="s11")
    # plt.legend()
    # plt.figure()
    # plt.plot(sp.get_frequency_list(), ideal3port.S(2,1,"PHASE"),label="s21")
    # plt.plot(sp.get_frequency_list(), ideal3port.S(1,1,"PHASE"),label="s11")
    # plt.legend()
    # plt.show()

    # aa=spfile()
    # aa.dosyaoku("./WG_EvanescentModeFilter Filtre_kapasiteli_loss_durumu1.s2p")
    # bb=spfile()
    # bb.dosyaoku("./WG_EvanescentModeFilter Filtre_kapasiteli_loss_durumu.s2p")
    # aa.dosyaoku("./5port_kapasiteli3.s5p")
    # a=0.016
    # b=0.0005
    # er=3.38
    # c1=0.564e-12
    # c2=0.836e-12
    # aa.change_ref_impedance_powerwave([lambda x:-0.5j/np.pi/x/c1*(1+0.01j),lambda x:-0.5j/np.pi/x/c2*(1+0.01j),lambda x:-0.5j/np.pi/x/c1*(1+0.01j),lambda x:Z_WG_TE10(er,a,b,x)*(1+0.01j),lambda x:Z_WG_TE10(er,a,b,x)*(1+0.01j)])
    # (data,adata,window,timeline,F_data)=aa.ImpulseResponse(i=2,j=1,dcinterp=1,dcvalue=(-200.0,0.0),FreqResCoef=10.0)
    # (bdata,badata,bwindow,btimeline,bF_data)=bb.ImpulseResponse(i=2,j=1,dcinterp=1,dcvalue=(-200.0,0.0),FreqResCoef=10.0)
    # from pylab import *
    #plot(window)
    # hold(True)
    #plot(abs(data)*window)
    # plot(timeline[1:],abs(F_data[1:]))
    # figure()
    # plot(btimeline[1:],abs(bF_data[1:]))
    #plot(abs(bdata)*bwindow)
    #plot(20.0*np.log10(abs(np.fft.fft(F_data))))
    # show()
