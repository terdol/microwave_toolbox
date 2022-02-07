# coding: utf-8
"""
Referans: A compact 2-D full-wave finite-difference frequency-domain method for general guided wave structures.pdf
A Compact 2-D FDFD Method for Modeling Microstrip Structures With Nonuniform Grids and Perfectly Matched Layer.pdf
Lecture 12 -- Finite-Difference Analysis of Waveguides (CEM Lectures) (Sadece matris boyutunu küçültmekte kullanıldı)
"""
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# from grid import CartesianGrid
from .grid2 import CartesianGrid
import scipy.sparse
from copy import deepcopy
from scipy.interpolate import griddata
import scipy.constants as sabitler
import matplotlib.colorbar as mcolorbar
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import inspect

nz = mcolors.Normalize()
co = sabitler.speed_of_light
epso = sabitler.epsilon_0
mu_0 = sabitler.mu_0
np.set_printoptions(precision=4)

def getindice(cc, dizi, pml=0):
    count = 0
    length = 0
    while (length < (cc * (1.0 - 1.0e-4))):
        try:
            length = length + dizi[count+pml]
        except:
            print("end of dizi")
            return count
        count = count + 1
    return count+pml


def exn(i, j):
    """ indice of Ex(i,j) in field vector """
    if i > (exn.Nx - 1) or i < 0:
        return None
    if j < 1 or j > (exn.Ny - 1):
        return None
    a = exn.Nx * (exn.Ny - 1)  # ExSayisi = HySayisi
    b = exn.Ny * (exn.Nx - 1) # EySayisi = HxSayisi
    return i + exn.Nx * (j - 1)


def eyn(i, j):
    """ indice of Ey(i,j) in field vector """
    if j > (eyn.Ny - 1) or j < 0:
        return None
    if i < 1 or i > (eyn.Nx - 1):
        return None
    a = eyn.Nx * (eyn.Ny - 1)  # ExSayisi = HySayisi
    b = eyn.Ny * (eyn.Nx - 1)  # EySayisi = HxSayisi
    return a + i - 1 + (eyn.Nx - 1) * j


def hxn(i, j):
    """ indice of Hx(i,j) in field vector """
    if j == hxn.Ny:
        return hxn(i, hxn.Ny - 1)
    elif j == -1:
        return hxn(i, 0)
    if j > hxn.Ny - 1 or j < 0:
        return None
    if i < 1 or i > hxn.Nx - 1:
        return None
    # a = hxn.Nx * (hxn.Ny - 1)  # ExSayisi = HySayisi
    # b = hxn.Ny * (hxn.Nx - 1)  # EySayisi = HxSayisi
    return i - 1 + (hxn.Nx - 1) * j


def hyn(i, j):
    """ indice of Hy(i,j) in field vector """
    if i == hyn.Nx:
        return hyn(hyn.Nx - 1, j)
    elif i == -1:
        return hyn(0, j)
    if i > hyn.Nx - 1 or i < 0:
        return None
    if j < 1 or j > hyn.Ny - 1:
        return None
    # a = hyn.Nx * (hyn.Ny - 1)  # ExSayisi = HySayisi
    b = hyn.Ny * (hyn.Nx - 1)  # EySayisi = HxSayisi
    return b + i + hyn.Nx * (j - 1)


def fdfd_mesh(boxwidth, boxheight, xres, yres, parts):
    """Create FDFD mesh.

    Args:
        boxwidth (float): Width of metal box.
        boxheight (float): Height of metal box.
        xres (float): Resolution along x-axis.
        yres (float): Resolution along y-axis
        parts (list): List of parts

    Returns:
        tuple: A tuple containing the following elements:

            1. dx, 1-d array containing the length of grid cells along x
            2. dy, 1-d array containing the length of grid cells along y
            3. er, dielectric permittivity 2-d array
            4. erxx, average dielectric permittivity 2-d array along x
            5. eryy, average dielectric permittivity 2-d array along y
            6. erzz, average dielectric permittivity 2-d array along z
            7. metalnodesx, indices of x-oriented grid edges
            8. metalnodesy, indices of y-oriented grid edges
            9. metalnodesz, indices of z-oriented grid edges
            10. currentloops, indices of grid edges used to calculate current
            11. vlines, Field component and cell indices that are used to calculate voltage
    """
    cg = CartesianGrid()
    kritikx = []
    kritiky = []

    kritikx.append(0.0)
    kritikx.append(boxwidth)
    kritiky.append(0.0)
    kritiky.append(boxheight)
    for part in parts:
        if part[0] == "dielectric":
            kritikx.append(part[2] - part[4]/ 2.0)
            kritikx.append(part[2] + part[4]/ 2.0)
            kritiky.append(part[3] - part[5]/ 2.0)
            kritiky.append(part[3] + part[5]/ 2.0)

        if part[0] == "metal":
            kritikx.append(part[2] - part[4]/ 2.0)
            kritikx.append(part[2] + part[4]/ 2.0)
            kritiky.append(part[3] - part[5]/ 2.0)
            kritiky.append(part[3] + part[5]/ 2.0)

    cg.findsubregions(kritikx, kritiky, [0])
    print("kritikx ", kritikx)
    print("kritiky ", kritiky)
    cg.nu_grid(xres, yres, 1.0)
    # if (inspect.stack()[1][3] == "microstrip" or inspect.stack()[1][3] == "stripline"):
    #     if (cg.get_nof_sub(1)==3):
    #         d1 = cg.return_subregion_params(1,1)[0]
    #         d2 = cg.return_subregion_params(1,2)[1]
    #         cg.customgrid(d1,d2,2, 1.2,0, 1,0)
    #         d1 = cg.return_subregion_params(1,1)[0]
    #         d2 = cg.return_subregion_params(1,0)[1]
    #         cg.customgrid(d1,d2,0, 1.2,1, 1,0)
    #         cg.customgrid(0.5*resolution, resolution, 1, 1.2,2, 0,0)
    #         cg.customgrid(0.5*resolution, resolution, 0, 1.2,1, 0,0)
    #         cg.customgrid(0.5*resolution, resolution, 2, 1.2,0, 0,0)
    # if (inspect.stack()[1][3] == "suspended"):
    #     if (cg.get_nof_sub(1)==4):
    #         d1 = cg.return_subregion_params(1,2)[0]
    #         d2 = cg.return_subregion_params(1,3)[1]
    #         cg.customgrid(d1,d2,3, 1.2,0, 1,0)
    #         d1 = cg.return_subregion_params(1,2)[0]
    #         d2 = cg.return_subregion_params(1,1)[1]
    #         cg.customgrid(d1,d2,1, 1.2,1, 1,0)
    #         cg.customgrid(0.5*resolution, resolution, 1, 1.2,2, 0,0)
    #         cg.customgrid(0.5*resolution, resolution, 0, 1.2,1, 0,0)
    #         cg.customgrid(0.5*resolution, resolution, 2, 1.2,0, 0,0)

    if (inspect.stack()[1][3] == "multilayer"):
        if cg.get_nof_sub(1) > len([part for part in parts if part[0] == "dielectric"]):
            for part in parts:
                if part[0] == "metal":
                    metal_y_subregion = cg.get_subregion_index_of_point(part[3], 1)
                    break
            print("metal_y_subregion = ", metal_y_subregion)
            d1 = cg.return_subregion_params(1, metal_y_subregion)[0]
            d2 = cg.return_subregion_params(1, metal_y_subregion + 1)[1]
            cg.customgrid(d1, d2, metal_y_subregion + 1, 1.2, 0, 1, 0)
            d1 = cg.return_subregion_params(1, metal_y_subregion)[0]
            d2 = cg.return_subregion_params(1, metal_y_subregion - 1)[1]
            cg.customgrid(d1, d2, metal_y_subregion - 1, 1.2, 1, 1, 0)
            cg.customgrid(0.5 * xres, yres, 1, 1.2, 2, 0, 0)
            cg.customgrid(0.5 * xres, yres, 0, 1.2, 1, 0, 0)
            cg.customgrid(0.5 * xres, yres, 2, 1.2, 0, 0, 0)

    dx = list(cg.dx)
    dy = list(cg.dy)
    print("dx= ", dx)
    print("dy= ", dy)
    Nx = len(dx)
    Ny = len(dy)
    exn.Nx = eyn.Nx = hxn.Nx = hyn.Nx = Nx
    exn.Ny = eyn.Ny = hxn.Ny = hyn.Ny = Ny
    # Hucrelerdeki maddelerin epsilonları
    er = np.ones((Nx, Ny))
    metalnodesx = []
    metalnodesy = []
    metalnodesz = []
    currentloops = []
    vlines = []


    for part in parts:
        if part[0] == "dielectric":
            xi1 = getindice(part[2] - (part[4]/ 2.0), dx)
            xi2 = getindice(part[2] + (part[4]/ 2.0), dx)
            yi1 = getindice(part[3] - (part[5]/ 2.0), dy)
            yi2 = getindice(part[3] + (part[5]/ 2.0), dy)
            kritikx.append(xi1)
            kritikx.append(xi2)
            kritiky.append(yi1)
            kritiky.append(yi2)
            for i in range(xi1, xi2):
                for j in range(yi1, yi2):
                    er[i, j] = part[6]

        if part[0] == "metal":
            #currentloop = []
            cloop = []
            xi1 = getindice(part[2] - (part[4]/ 2.0), dx)
            xi2 = getindice(part[2] + (part[4]/ 2.0), dx)
            yi1 = getindice(part[3] - (part[5]/ 2.0), dy)
            yi2 = getindice(part[3] + (part[5]/ 2.0), dy)
            kritikx.append(xi1)
            kritikx.append(xi2)
            kritiky.append(yi1)
            kritiky.append(yi2)
            cc = getindice((part[2]), dx)
            # vline = [(eyn(cc, i), dy[i]) for i in range(yi1)]
            vlines.append(("eyn",cc, 0, yi1))
            for j in range(yi1, yi2 + 1):
                # currentloop.append((hyn(xi2, j), -0.5 * (dy[j] + dy[j - 1])))
                # currentloop.append((hyn(xi1 - 1, j), 0.5 * (dy[j] + dy[j - 1])))
                if j==0:
                    cloop.append(("hyn",xi2, j, -0.5 * (dy[j])))
                    cloop.append(("hyn",xi1 - 1, j, 0.5 * (dy[j])))
                elif j==len(dy):
                    cloop.append(("hyn",xi2, j, -0.5 * (dy[j - 1])))
                    cloop.append(("hyn",xi1 - 1, j, 0.5 * (dy[j - 1])))
                else:
                    cloop.append(("hyn",xi2, j, -0.5 * (dy[j] + dy[j - 1])))
                    cloop.append(("hyn",xi1 - 1, j, 0.5 * (dy[j] + dy[j - 1])))
            for i in range(xi1, xi2 + 1):
                # currentloop.append((hxn(i, yi2), 0.5 * (dx[i] + dx[i - 1])))
                # currentloop.append((hxn(i, yi1 - 1), -0.5 * (dx[i] + dx[i - 1])))
                if i == 0:
                    cloop.append(("hxn",i, yi2, 0.5 * (dx[i])))
                    cloop.append(("hxn",i, yi1 - 1, -0.5 * (dx[i])))
                elif i ==len(dx):
                    cloop.append(("hxn",i, yi2, 0.5 * (dx[i - 1])))
                    cloop.append(("hxn",i, yi1 - 1, -0.5 * (dx[i - 1])))
                else:
                    cloop.append(("hxn",i, yi2, 0.5 * (dx[i] + dx[i - 1])))
                    cloop.append(("hxn",i, yi1 - 1, -0.5 * (dx[i] + dx[i - 1])))
                for j in range(yi1, yi2 + 1):
                    if j>0 and j<len(dy):
                        if i>0 and i<len(dx):
                            metalnodesz.append((i, j))
                    if i < (xi2):
                        if j>0 and j<len(dy):
                            if i>0 and i<len(dx):
                                metalnodesx.append((i, j))
                    if j < (yi2):
                        if j>0 and j<len(dy):
                            if i>0 and i<len(dx):
                                metalnodesy.append((i, j))
            # currentloops.append(currentloop)
            currentloops.append(cloop)

    # E-Fieldların hissettiği epsilon değerleri
    erxx = np.matrix(np.ones((Nx, Ny + 1), float))
    eryy = np.matrix(np.ones((Nx + 1, Ny), float))
    erzz = np.matrix(np.ones((Nx + 1, Ny + 1), float))

    for i in range(Nx):
        erxx[i, 0] = er[i, 0]
        erxx[i, Ny] = er[i, Ny - 1]
        for j in range(1, Ny):
            erxx[i, j] = 0.5 * (er[i, j - 1] + er[i, j])

    for j in range(Ny):
        eryy[0, j] = er[0, j]
        eryy[Nx, j] = er[Nx - 1, j]
        for i in range(1, Nx):
            eryy[i, j] = 0.5 * (er[i - 1, j] + er[i, j])

    for i in range(1, Nx):
        for j in range(1, Ny):
            erzz[i, j] = 0.25 * (er[i, j] + er[i, j - 1] + er[i - 1, j] + er[i - 1, j - 1])
        erzz[i, 0] = 0.5 * (er[i, 0] + er[i - 1, 0])
        erzz[i, Ny] = 0.5 * (er[i, Ny - 1] + er[i - 1, Ny - 1])
    for j in range(1, Ny):
        erzz[0, j] = 0.5 * (er[0, j] + er[0, j - 1])
        erzz[Nx, j] = 0.5 * (er[Nx - 1, j] + er[Nx - 1, j - 1])

    erzz[0, 0] = er[0, 0]
    erzz[0, Ny] = er[0, Ny - 1]
    erzz[Nx, Ny] = er[Nx - 1, Ny - 1]
    erzz[Nx, 0] = er[Nx - 1, 0]
    return (dx, dy, er, erxx, eryy, erzz, metalnodesx, metalnodesy, metalnodesz, currentloops, vlines)

def addpml(pmllayers,dx, dy, er, erxx, eryy, erzz, metalnodesx, metalnodesy, metalnodesz, currentloops, vlines):
    """Add PML (perfectly matched layers) and return modified grid and permittivity arrays.

    Args:
        pmllayers (list of int): Number of PML layers x1,x2,y1,y2.
        dx (list of float): lengths of x-oriented grid cells
        dy (list of float): lengths of y-oriented grid cells
        er (2-d array of float): Dielectric permittivity array
        erxx (2-d array of float): Average dielectric permittivities at x-oriented grid edges
        eryy (2-d array of float): Average dielectric permittivities at y-oriented grid edges
        erzz (2-d array of float): Average dielectric permittivities at z-oriented grid edges
        metalnodesx (list of 2-tuples of int): indices of x-oriented grid edges
        metalnodesy (list of 2-tuples of int): indices of y-oriented grid edges
        metalnodesz (list of 2-tuples of int): indices of z-oriented grid edges
        currentloops (list): Indices of grid edges used to calculate current
        vlines (list): Field component and cell indices that are used to calculate voltage

    Returns:
        [type]: [description]
    """
    x1,x2,y1,y2=tuple(pmllayers)
    yenidx=[dx[0]]*x1+dx+[dx[-1]]*x2
    yenidy=[dy[0]]*y1+dy+[dy[-1]]*y2
    Nx=len(yenidx)
    Ny=len(yenidy)
    print("Nx, Ny ",Nx,Ny)
    exn.Nx = eyn.Nx = hxn.Nx = hyn.Nx = Nx
    exn.Ny = eyn.Ny = hxn.Ny = hyn.Ny = Ny

    newerxx = np.matrix(np.ones((Nx, Ny + 1), float))
    neweryy = np.matrix(np.ones((Nx + 1, Ny), float))
    newerzz = np.matrix(np.ones((Nx + 1, Ny + 1), float))
    newer = np.matrix(np.ones((Nx, Ny), float))
    newerxx[:x1,:y1]=erxx[0,0]
    neweryy[:x1,:y1]=eryy[0,0]
    newerzz[:x1,:y1]=erzz[0,0]
    newer[:x1,:y1]=er[0,0]
    newerxx[:x1,-y2:]=erxx[0,-1]
    neweryy[:x1,-y2:]=eryy[0,-1]
    newerzz[:x1,-y2:]=erzz[0,-1]
    newer[:x1,-y2:]=er[0,-1]
    newerxx[-x2:,:y1]=erxx[-1,0]
    neweryy[-x2:,:y1]=eryy[-1,0]
    newerzz[-x2:,:y1]=erzz[-1,0]
    newer[-x2:,:y1]=er[-1,0]
    newerxx[-x2:,-y2:]=erxx[-1,-1]
    neweryy[-x2:,-y2:]=eryy[-1,-1]
    newerzz[-x2:,-y2:]=erzz[-1,-1]
    newer[-x2:,-y2:]=er[-1,-1]
    for i in range(x1,Nx-x2):
        newerxx[i, -y2:] = erxx[i-x1, -1]
        neweryy[i, -y2:] = eryy[i-x1, -1]
        newerzz[i, -y2:] = erzz[i-x1, -1]
        newer[i, -y2:] = er[i-x1, -1]
        newerxx[i, :y1] = erxx[i-x1, 0]
        neweryy[i, :y1] = eryy[i-x1, 0]
        newerzz[i, :y1] = erzz[i-x1, 0]
        newer[i, :y1] = er[i-x1, 0]
    for j in range(y1,Ny-y2):
        newerxx[-x2:, j] = erxx[-1, j-y1]
        neweryy[-(x2+1):, j] = eryy[-1, j-y1]
        newerzz[-(x2+1):, j] = erzz[-1, j-y1]
        newer[-x2:, j] = er[-1, j-y1]
        newerxx[:x1, j] = erxx[0, j-y1]
        neweryy[:x1, j] = eryy[0, j-y1]
        newerzz[:x1, j] = erzz[0, j-y1]
        newer[:x1, j] = er[0, j-y1]
    for i in range(x1,Nx-x2):
        for j in range(y1,Ny-y2):
            newerxx[i, j] = erxx[i-x1, j-y1]
            neweryy[i, j] = eryy[i-x1, j-y1]
            newerzz[i, j] = erzz[i-x1, j-y1]
            newer[i, j] = er[i-x1, j-y1]

    newmetalnodesx=[(x1+i,y1+j) for (i,j) in metalnodesx]
    newmetalnodesy=[(x1+i,y1+j) for (i,j) in metalnodesy]
    newmetalnodesz=[(x1+i,y1+j) for (i,j) in metalnodesz]
    newcurrentloops=[[(a,x1+i,y1+j,b) for (a,i,j,b) in cloops] for cloops in currentloops]
    newvlines=[(a,x1+i,y1+j1,y1+j2) for (a,i,j1,j2) in vlines]

    print("newer= ", newer)
    # print("erxx= ", erxx)
    # print("eryy= ", eryy)
    # print("erzz= ", erzz)
    print("newerxx= ", newerxx)
    print("neweryy= ", neweryy)
    print("newerzz= ", newerzz)
    return (yenidx, yenidy, newer, newerxx, neweryy, newerzz, newmetalnodesx, newmetalnodesy, newmetalnodesz, newcurrentloops, newvlines)

def sigma(i,dx,nlayer):
    # dx : dizi, hucre boyutlari
    # nlayer : tuple, bastan ve sondan pml layer sayisi
    Rth=1.0e-7
    n=2.0
    n1=nlayer[0]
    n2=nlayer[1]
    if i>=n1 and i<=(len(dx)-n2):
        return 0
    elif i>=len(dx):
        return 0
    elif i<n1:
        delta1=sum(dx[:n1])
        dd=sum(dx[:int(i)])+(i-int(i))*dx[int(i)]
        return -(n+1)*epso*co/2.0/delta1*np.log(Rth)*np.power((delta1-dd)/delta1,n)
    elif i > (len(dx)-n2):
        delta2 = sum(dx[-n2:])
        dd = sum(dx[(len(dx)-n2):int(i)]) + (i - int(i)) * dx[int(i)]
        return -(n + 1) * epso * co / 2.0 / delta2 * np.log(Rth) * np.power(dd/ delta2, n)
def sx(i):
    """
    sigma.arr : dizi, hucre boyutlari
    sigma.nlayer : list, bastan ve sondan pml layer sayisi
    """
    return sigma(i,sx.arr,sx.nlayer)
def sy(i):
    """
    sigma.arr : dizi, hucre boyutlari
    sigma.nlayer : list, bastan ve sondan pml layer sayisi
    """
    return sigma(i,sy.arr,sy.nlayer)
def s_0(i):
    return 0
def fdfd_solve(dx, dy, er, erxx, eryy, erzz, metalnodesx, metalnodesy, metalnodesz, cloops, vlines, freq, pmllayers,sparse = True):

    sy.arr = dy
    sx.arr = dx
    sx.nlayer = pmllayers[:2]
    sy.nlayer = pmllayers[-2:]
    import scipy.sparse
    print("metalnodesx ",metalnodesx)
    print("metalnodesy ",metalnodesy)
    print("metalnodesz ",metalnodesz)
    print("cloops ",cloops)
    # print("vline ",vline)

    ko = 2 * np.pi * freq / sabitler.speed_of_light

    # condition number düzeltme (bu yöntemin yanlış olduğuna karar verdim)
    #dd=np.min([np.min(dx),np.min(dy)])
    #coe = (1.0/ko/dd)/500
    #ko = ko*coe

    Nx = len(dx)
    Ny = len(dy)
    a = Nx * (Ny - 1)  # ExSayisi = HySayisi
    b = Ny * (Nx - 1)  # EySayisi = HxSayisi
    #sparse= True

    if sparse:
        P = scipy.sparse.lil_matrix((a + b, a + b), dtype=np.complex128)
        Q = scipy.sparse.lil_matrix((a + b, a + b), dtype=np.complex128)
        # A = scipy.sparse.lil_matrix((2 * a + 2 * b, 2 * a + 2 * b), dtype=np.complex128)
        M1 = scipy.sparse.lil_matrix((a + b, a + b), dtype=np.complex128)
        # M1 = scipy.sparse.lil_matrix((2 * a + 2 * b, 2 * a + 2 * b), dtype=np.complex128)
    else:
        P = np.matlib.zeros((a + b, a + b), dtype=np.complex128)
        Q = np.matlib.zeros((a + b, a + b), dtype=np.complex128)
        # A = np.zeros((2 * a + 2 * b, 2 * a + 2 * b), dtype=np.complex128)
        M1 = np.matlib.zeros((a + b, a + b), dtype=np.complex128)
        # M1 = np.zeros((2 * a + 2 * b, 2 * a + 2 * b), dtype=np.complex128)

    dx = dx + [dx[-1]] + [dx[0]]  # bu sayede dx[-1] dx[0]'a esit olacak ve dy[Nx], dy[Nx-1]'e esit olacak.
    dy = dy + [dy[-1]] + [dy[0]]  # bu sayede dy[-1] dy[0]'a esit olacak ve dy[Ny], dy[Ny-1]'e esit olacak.
    ddx = [0.5 * (dx[i] + dx[i - 1]) for i in range(1, len(dx))]
    ddy = [0.5 * (dy[i] + dy[i - 1]) for i in range(1, len(dy))]
    dx = np.array(dx)
    dy = np.array(dy)

    print("dx= ", dx)
    print("dy= ", dy)
    print("ddx= ", ddx)
    print("ddy= ", ddy)

    # np.savetxt("erxx.txt", erxx)
    # np.savetxt("eryy.txt", eryy)
    # np.savetxt("erzz.txt", erzz)
    for i in range(a + b):
        M1[i, i] = 1.0
    # matrisdosya = open("MAT.txt","w")
    # mdosya = open("MATM.txt","w")
    print("2*a+2*b= ",2*a+2*b)
    print("Nx= ",Nx)
    print("Ny= ",Ny)
    # def AF(x, y, z):
        # if y is not None:
            # A[x, y] = A[x, y] + z
    def PF(x, y, z):
        if y is not None:
            P[x, y] = P[x, y] + z
    def QF(x, y, z):
        if y is not None:
            Q[x, y] = Q[x, y] + z

    exn.Nx = eyn.Nx = hxn.Nx = hyn.Nx = Nx
    exn.Ny = eyn.Ny = hxn.Ny = hyn.Ny = Ny
    for i in range(Nx):
        for j in range(1, Ny):
            row = exn(i, j)
            cc = c1 = c2 = c3 = c4 = 1.0
            if i>0:
                c3 = (1.0/(1.0 - 1.0j * sx(i) / (ko * co * epso * 0.5 * (erxx[i, j] + erxx[i - 1, j]))))
            if i<Nx-1:
                c1 = (1.0/ (1.0 - 1.0j * sx(i + 1.0) / (ko * co * epso * 0.5 * (erxx[i + 1, j] + erxx[i, j]))))
                c2 = (1.0/ (1.0 - 1.0j * sy(j) / (ko * co * epso * 0.5 * (eryy[i + 1, j] + eryy[i + 1, j - 1]))))
            cc = (1.0/(1.0-1.0j*sx(i+0.5)/(ko*co*epso*erxx[i,j])))
            c4 = (1.0/(1.0-1.0j*sy(j)/(ko*co*epso*0.5*(eryy[i,j]+eryy[i,j-1]))))
            if row is not None:
                if i > 0 and ((i, j) not in metalnodesz):  # Ez(i,j) katsayilari
                    PF(row, hxn(i, j - 1), -1.0 / erzz[i, j] / ko ** 2 / dx[i] / ddy[j - 1] *cc*c4)
                    PF(row, hxn(i, j), 1.0 / erzz[i, j] / ko ** 2 / dx[i] / ddy[j - 1] *cc*c4)
                    PF(row, hyn(i - 1, j), 1.0 / erzz[i, j] / ko ** 2 / dx[i] / ddx[i - 1] *cc*c3)
                    PF(row, hyn(i, j), -(1.0 / ddx[i - 1] / erzz[i, j]) / ko ** 2 / dx[i] *cc*c3)

                if i < Nx - 1 and ((i + 1, j) not in metalnodesz):  # Ez(i+1,j) katsayilari
                    PF(row, hxn(i + 1, j), -1.0 / erzz[i + 1, j] / ko ** 2 / dx[i] / ddy[j - 1] *cc*c2)
                    PF(row, hxn(i + 1, j - 1), 1.0 / erzz[i + 1, j] / ko ** 2 / dx[i] / ddy[j - 1] *cc*c2)
                    PF(row, hyn(i + 1, j), 1.0 / erzz[i + 1, j] / ko ** 2 / dx[i] / ddx[i] *cc*c1)
                    PF(row, hyn(i, j), -(1.0 / ddx[i] / erzz[i + 1, j]) / ko ** 2 / dx[i] *cc*c1)
                PF(row, hyn(i, j), 1.0)

    for i in range(1, Nx):
        for j in range(Ny):
            row = eyn(i, j)
            cc = c1 = c2 = c3 = c4 = 1.0
            if j > 0:
                c4 = (1.0/ (1.0 - 1.0j * sy(j) / (ko * co * epso * 0.5 * (eryy[i, j] + eryy[i, j - 1]))))
            if j < Ny-1:
                c1 = (1.0/ (1.0 - 1.0j * sx(i) / (ko * co * epso * 0.5 * (erxx[i, j+1] + erxx[i-1, j+1]))))
                c2 = (1.0/ (1.0 - 1.0j * sy(j+1) / (ko * co * epso * 0.5 * (eryy[i, j+1] + eryy[i, j]))))
            cc = (1.0/ (1.0 - 1.0j * sy(j + 0.5) / (ko * co * epso * eryy[i, j])))
            c3 = (1.0/ (1.0 - 1.0j * sx(i) / (ko * co * epso * 0.5 * (erxx[i, j] + erxx[i - 1, j]))))
            if row is not None:
                if j > 0 and ((i, j) not in metalnodesz):  # Ez(i,j) katsayilari
                    PF(row, hyn(i - 1, j), 1.0 / erzz[i, j] / ko ** 2 / ddx[i - 1] / dy[j] *cc*c3)
                    PF(row, hyn(i, j), -1.0 / erzz[i, j] / ko ** 2 / ddx[i - 1] / dy[j] *cc*c3)
                    PF(row, hxn(i, j - 1), -1.0 / erzz[i, j] / ko ** 2 / ddy[j - 1] / dy[j] *cc*c4)
                    PF(row, hxn(i, j), (1.0 / ddy[j - 1] / erzz[i, j]) / ko ** 2 / dy[j] *cc*c4)
                if j < Ny - 1 and ((i, j + 1) not in metalnodesz):  # Ez(i,j+1) katsayilari
                    PF(row, hyn(i - 1, j + 1), -1.0 / erzz[i, j + 1] / ko ** 2 / ddx[i - 1] / dy[j] *cc*c1)
                    PF(row, hyn(i, j + 1), 1.0 / erzz[i, j + 1] / ko ** 2 / ddx[i - 1] / dy[j] *cc*c1)
                    PF(row, hxn(i, j + 1), -1.0 / erzz[i, j + 1] / ko ** 2 / ddy[j] / dy[j] *cc*c2)
                    PF(row, hxn(i, j), (1.0 / ddy[j] / erzz[i, j + 1]) / ko ** 2 / dy[j] *cc*c2)
                PF(row, hxn(i, j), -1.0)

    for i in range(Nx):
        for j in range(Ny):
            row = hyn(i, j)
            cc = c1 = c2 = c3 = c4 = 1.0
            if j > 0:
                c4 = (1.0/ (1.0 - 1.0j * sy(j - 0.5) / (ko * co * epso * er[i, j-1])))
                cc = (1.0/ (1.0 - 1.0j * sy(j) / (ko * co * epso * 0.5 * (eryy[i, j]+eryy[i, j-1]))))
                c3 = (1.0/ (1.0 - 1.0j * sx(i + 0.5) / (ko * co * epso * er[i, j-1])))
            c1 = (1.0/ (1.0 - 1.0j * sx(i + 0.5) / (ko * co * epso * er[i, j])))
            c2 = (1.0/ (1.0 - 1.0j * sy(j + 0.5) / (ko * co * epso * er[i, j])))
            if row is not None:
                QF(row, eyn(i, j - 1), -1.0 / ko ** 2 / dx[i] / (0.5 * (dy[j] + dy[j - 1])) *cc*c3)
                QF(row, eyn(i, j), 1.0 / ko ** 2 / dx[i] / (0.5 * (dy[j] + dy[j - 1])) *cc*c1)
                QF(row, eyn(i + 1, j), -1.0 / ko ** 2 / dx[i] / (0.5 * (dy[j] + dy[j - 1])) *cc*c1)
                QF(row, eyn(i + 1, j - 1), 1.0 / ko ** 2 / dx[i] / (0.5 * (dy[j] + dy[j - 1])) *cc*c3)
                QF(row, exn(i, j - 1), 1.0 / ko ** 2 / dy[j - 1] / (0.5 * (dy[j] + dy[j - 1])) *cc*c4)
                QF(row, exn(i, j + 1), 1.0 / ko ** 2 / dy[j] / (0.5 * (dy[j] + dy[j - 1])) *cc*c2)
                QF(row, exn(i, j),
                   erxx[i, j] - (1.0 / dy[j] *cc*c2 + 1.0 / (dy[j - 1]) *cc*c4) / ko ** 2 / (0.5 * (dy[j] + dy[j - 1])))

    for i in range(Nx):
        for j in range(Ny):
            row = hxn(i, j)
            cc = c1 = c2 = c3 = c4 = 1.0
            if i > 0:
                c4 = (1.0/ (1.0 - 1.0j * sy(j + 0.5) / (ko * co * epso * er[i-1, j])))
                cc = (1.0/ (1.0 - 1.0j * sx(i) / (ko * co * epso * 0.5 * (erxx[i, j]+erxx[i-1, j]))))
                c3 = (1.0/ (1.0 - 1.0j * sx(i - 0.5) / (ko * co * epso * er[i-1, j])))
            c1 = (1.0/ (1.0 - 1.0j * sx(i + 0.5) / (ko * co * epso * er[i, j])))
            c2 = (1.0/ (1.0 - 1.0j * sy(j + 0.5) / (ko * co * epso * er[i, j])))
            if row is not None:
                QF(row, exn(i - 1, j), 1.0 / ko ** 2 / (0.5 * (dx[i] + dx[i - 1])) / dy[j] *cc*c4)
                QF(row, exn(i, j), -1.0 / ko ** 2 / (0.5 * (dx[i] + dx[i - 1])) / dy[j] *cc*c2)
                QF(row, exn(i - 1, j + 1), -1.0 / ko ** 2 / (0.5 * (dx[i] + dx[i - 1])) / dy[j] *cc*c4)
                QF(row, exn(i, j + 1), 1.0 / ko ** 2 / (0.5 * (dx[i] + dx[i - 1])) / dy[j] *cc*c2)
                QF(row, eyn(i - 1, j), -1.0 / ko ** 2 / (0.5 * (dx[i] + dx[i - 1])) / dx[i - 1] *cc*c3)
                QF(row, eyn(i + 1, j), -1.0 / ko ** 2 / (0.5 * (dx[i] + dx[i - 1])) / dx[i] *cc*c1)
                QF(row, eyn(i, j),
                   -eryy[i, j] + (1.0 / dx[i] *cc*c1 + 1.0 / dx[i - 1] *cc*c3) / ko ** 2 / (0.5 * (dx[i] + dx[i - 1])))

    # plt.imshow(np.abs(np.matrix(A)))
    # plt.title("A")
    # plt.show()
    # plt.plot([np.abs(x) for x in A[0,:]],label="0")
    # plt.plot([np.abs(x) for x in A[1,:]],label="1")
    # plt.show()
    coef=1e30
    for i, j in metalnodesx:
        r = exn(i, j)
        P[r, r] = P[r, r] * coef
        # Q[r, r] = Q[r, r] * coef
        M1[r, r] = M1[r, r] * coef
        # mdosya.write(str(r)+"\n")
    for i, j in metalnodesy:
        r = eyn(i, j)
        P[r, r] = P[r, r] * coef
        # Q[r, r] = Q[r, r] * coef
        M1[r, r] = M1[r, r] * coef
        # mdosya.write(str(r)+"\n")
    # matrisdosya.close()
    # mdosya.close()
    # for i in range(2 * a + 2 * b):
    #     for j in range(2 * a + 2 * b):
    #         A[i, j] = A[i, j]/100.0
    #         M1[i, j] = M1[i, j]/100.0
    import scipy.sparse.linalg
    from numpy import linalg as LA
    import scipy.linalg
    # A = A + coe*np.eye(2 * a + 2 * b)
    if sparse:
        la, v = scipy.sparse.linalg.eigs(P*Q, k=100, M=M1, sigma=np.mean(er), tol=0.001, which="LM")
    else:
        cond_number=np.linalg.cond(P*Q)
        print("condition number= ",cond_number)
        la, v = scipy.linalg.eig(P*Q, M1)
    # Field Vector: [Ex Ey]
    # eigenvalue, eigenvector pairs
    eigs = []
    for i in range(len(la)):
        print("la[i] ",la[i])
        if abs(la[i].imag / la[i].real) < 0.02 and la[i].real > 0.1  and la[i].real < 2*np.max(er):
        # if abs(la[i].imag) < 1e-2 and la[i].real > 0.1 and abs(la[i].real)**2 < np.max(er)*1.2:
            print("in_la[i] ",la[i])
            eigs.append((la[i], v[:, i]))

    output = []

    # plt.imshow(erxx)
    # plt.title("erxx")
    # plt.show()
    # plt.imshow(eryy)
    # plt.title("eryy")
    # plt.show()

    #TE10 deneme
    # EV = np.zeros((2 * a + 2 * b), dtype=np.complex128)
    # boxwidth = 0.012  # WG genişliği
    # boxheight = 0.03  # WG yüksekliği
    # dielheight = 0.009
    # metalwidth = 0.0095
    # metalthickness = 0.001
    # freq = 8e9
    # epsilon = 1.0
    # pmllayers=[0,0,0,0]
    # ko = 2.0 * np.pi * freq / co
    # kc = np.pi/boxwidth
    # omega = 2.0*np.pi*freq
    # beta = np.sqrt(ko*ko-kc*kc)
    # for i in range(Nx+1):
        # for j in range(Ny+1):
            # x=np.sum(dx[:i])
            # y=np.sum(dy[:j])
            # print "i,j ",i,j,eyn(i,j),hxn(i,j)
            # if eyn(i,j) is not None:
                # EV[eyn(i,j)]=-1.0j*omega*mu_0/(kc)**2*np.sin(np.pi*x/boxwidth) / np.sqrt(mu_0/epso)
            # x=np.sum(dx[:i])
            # y=np.sum(dy[:j])
            # if hxn(i,j) is not None:
                # EV[hxn(i,j)]=1.0j*beta/(kc)**2*np.sin(np.pi*x/boxwidth) #* np.sqrt(mu_0/epso)
    # for i in range(Nx+1):
        # for j in range(Ny+1):
            # if eyn(i,j) is not None:
                # print "dd ",EV[hxn(i,j)]/EV[eyn(i,j)], beta/ko,omega*mu_0/ np.sqrt(mu_0/epso),ko

    # XX = A.dot(EV)-(beta/ko)*EV
    # XX = np.dot(A, EV)-(beta/ko)*EV
    # XX = -(beta/ko)*EV
    # plt.plot(np.abs(XX))
    # XX = np.dot(A, EV)
    # plt.plot(np.abs(XX))
    # plt.plot(np.abs(EV))
    # print eigs[0][0]
    # plt.imshow(np.abs(A))
    # plt.plot(np.abs(eigs[0][1][:Nx*(Ny-1):Nx]))
    # plt.plot(np.abs(eigs[0][1]))
    # plt.plot(np.abs(EV))
    # plt.show()
    # print "A00 ",A[0,0],A[0,1],A[1,0],A[1,1]
    # plt.spy(np.abs(A))



    for la, Ev in eigs:
        currents = []
        voltages = []
        ko = 2 * np.pi * freq / co
        eeff = la
        lambda_g = 2 * np.pi / eeff / ko
        Hv = Q*np.matrix(Ev).transpose()
        Hv,=np.array(Hv.T)
        for cloop in cloops:
            print("cloop= ",cloop)
            temp=[(eval("{}({},{})".format(func,i,j)),dl) for func, i, j, dl in cloop]
            cf = [Hv[i]*j for i,j in temp if i is not None]
            currents.append( np.sum(cf))
        for cc in range(len(vlines)):
            func, i, j1, j2 = vlines[cc]
            print("vline= ",i,j1,j2)
            temp=[(eval("{}({},{})".format(func,i,j)),dy[j]) for j in range(j1,j2)]
            voltages.append( np.sum([v[i]*j for i,j in temp if i is not None]))
        Z = np.abs(np.sqrt((sabitler.mu_0/ sabitler.epsilon_0)) * np.array(voltages) / np.array(currents))
        print("eeff ",eeff)

        output.append((Z, eeff, lambda_g, la, (Ev,Hv) ))
    return output


def microstrip(boxwidth, boxheight, dielheight, metalwidth, metalthickness, epsilon, freq, xres, yres):
    resolution = np.min((metalwidth/ 10.0), co / freq / np.sqrt(epsilon) / 20.0)
    if yres == None:
        yres = resolution
    if xres == None:
        xres = resolution
    parts = []
    parts.append(("dielectric", "rect", (boxwidth/2.0), (dielheight/2.0), boxwidth, dielheight, epsilon))
    parts.append(("metal", "rect", (boxwidth/2.0), dielheight + (metalthickness/ 2.0), metalwidth, metalthickness))
    (dx, dy, er, erxx, eryy, erzz, metalnodesx, metalnodesy, metalnodesz, cloops) = fdfd_mesh(boxwidth, boxheight,
                                                                                              xres,yres, parts)
    cc = getindice((boxwidth/2.0), dx)
    print("cc= ", cc, getindice(dielheight, dy))
    vline = [(eyn(cc, i), dy[i]) for i in range(getindice(dielheight, dy))]
    return (dx, dy, er, erxx, eryy, erzz, metalnodesx, metalnodesy, metalnodesz, cloops, vline, parts)


def stripline(boxwidth, boxheight, metalwidth, metalthickness, epsilon, freq, xres, yres):
    resolution = np.min((metalwidth/10.0), co / freq / np.sqrt(epsilon) / 20.0)
    if yres == None:
        yres = resolution
    if xres == None:
        xres = resolution
    parts = []
    parts.append(("dielectric", "rect", (boxwidth/2.0), (boxheight/2.0), boxwidth, boxheight, epsilon))
    parts.append(("metal", "rect", (boxwidth/2.0), (boxheight/2.0), metalwidth, metalthickness))
    (dx, dy, er, erxx, eryy, erzz, metalnodesx, metalnodesy, metalnodesz, cloops) = fdfd_mesh(boxwidth, boxheight,
                                                                                              xres,yres, parts)
    cc = getindice((boxwidth/ 2.0), dx)
    print("cc= ", cc, getindice((boxheight/ 2.0) - (metalthickness/ 2.0), dy))
    vline = [(eyn(cc, i), dy[i]) for i in range(getindice((boxheight/ 2.0) - (metalthickness/ 2.0), dy))]
    return (dx, dy, er, erxx, eryy, erzz, metalnodesx, metalnodesy, metalnodesz, cloops, vline, parts)


def suspended(boxwidth, boxheight, dielthickness, metalheight, metalwidth, metalthickness, epsilon, freq, xres, yres):
    resolution = np.min((metalwidth/ 10.0), co / freq / np.sqrt(np.max(epsilon)) / 20.0)
    if yres == None:
        yres = resolution
    if xres == None:
        xres = resolution
    parts = []
    parts.append(
            ("dielectric", "rect", (boxwidth/ 2.0), metalheight - (dielthickness/ 2.0), boxwidth, dielthickness, epsilon))
    parts.append(("metal", "rect", (boxwidth/ 2.0), metalheight + (metalthickness/ 2.0), metalwidth, metalthickness))
    (dx, dy, er, erxx, eryy, erzz, metalnodesx, metalnodesy, metalnodesz, cloops) = fdfd_mesh(boxwidth, boxheight,
                                                                                               xres,yres, parts)
    cc = getindice((boxwidth/ 2.0), dx)
    print("cc= ", cc, getindice(metalheight, dy))
    vline = [(eyn(cc, i), dy[i]) for i in range(getindice((boxheight/ 2.0) - (metalthickness/2.0), dy))]
    return (dx, dy, er, erxx, eryy, erzz, metalnodesx, metalnodesy, metalnodesz, cloops, vline, parts)


def multilayer(boxwidth, boxheight, dielthickness, epsilon, metals, freq, pmllayers, xres, yres):

    """
    :type dielthickness: tabandan başlayarak yukarı dogru sirasiyla dielektriklerin kalinliklarinin listesi
    :type epsilon: tabandan başlayarak yukarı dogru sirasiyla dielektriklerin katsayilarinin listesi
    :type metals:şu elemanlardan oluşan dizi [metalheight, metalwidth, metalthickness, metaloffset]
        metalheight: metalin alt yüzeyinin yüksekliği
    """

    resolution = boxwidth/50.0
    print("yres= "+str(yres))
    if yres == None:
        yres = resolution
    if xres == None:
        xres = resolution
    parts = []
    temp = 0
    for diel, eps in zip(dielthickness, epsilon):
        parts.append(("dielectric", "rect", (boxwidth/ 2.0), temp + (diel/ 2.0), boxwidth, diel, eps))
        temp = temp + diel
    for metal in metals:
        print(metal)
        metalheight, metalwidth, metalthickness, metaloffset = tuple(metal)
        parts.append(("metal", "rect", (boxwidth/2.0)+metaloffset, metalheight + (metalthickness/ 2.0), metalwidth, metalthickness))
    (dx, dy, er, erxx, eryy, erzz, metalnodesx, metalnodesy, metalnodesz, cloops, vlines) = fdfd_mesh(boxwidth, boxheight,
                                                                                               xres,yres, parts)
    (dx, dy, er, erxx, eryy, erzz, metalnodesx, metalnodesy, metalnodesz, cloops, vlines) = addpml(pmllayers,dx, dy, er, erxx, eryy, erzz, metalnodesx, metalnodesy, metalnodesz, cloops, vlines)
    cc = getindice((boxwidth/ 2.0), dx, pmllayers[0])
    # print("cc= ", cc, getindice(metalheight, dy, pmllayers[2]))
    # vline = [(eyn(cc, i), dy[i]) for i in range(getindice(metalheight, dy, pmllayers[2]))]
    return (dx, dy, er, erxx, eryy, erzz, metalnodesx, metalnodesy, metalnodesz, cloops, vlines, parts)

def waveguide(boxwidth, boxheight, dielthickness, metalheight, metalwidth, metalthickness, epsilon, freq, pmllayers, xres, yres):
    """
    :type dielthickness: tabandan başlayarak yukarı dogru sirasiyla dielektriklerin kalinliklarinin listesi
    :type epsilon: tabandan başlayarak yukarı dogru sirasiyla dielektriklerin katsayilarinin listesi
    :type metalwidth: object
    """
    # resolution = np.min([(metalwidth/ 10.0), co / freq / np.sqrt(np.max(epsilon)) / 20.0, (np.min([boxheight,boxwidth])/40.0)])
    # resolution = (metalthickness/ 2.0)
    if yres == None:
        yres = resolution
    if xres == None:
        xres = resolution
    parts = []
    temp = 0
    # for diel, eps in zip(dielthickness, epsilon):
        # parts.append(("dielectric", "rect", boxwidth / 2.0, temp + diel / 2.0, boxwidth, diel, eps))
        # temp = temp + diel
    # parts.append(("metal", "rect", boxwidth / 2.0, metalheight + metalthickness / 2.0, metalwidth, metalthickness))
    (dx, dy, er, erxx, eryy, erzz, metalnodesx, metalnodesy, metalnodesz, cloops, vlines) = fdfd_mesh(boxwidth, boxheight,
                                                                                               xres,yres, parts)
    (dx, dy, er, erxx, eryy, erzz, metalnodesx, metalnodesy, metalnodesz, cloops, vlines) = addpml(pmllayers,dx, dy, er, erxx, eryy, erzz, metalnodesx, metalnodesy, metalnodesz, cloops, vlines)
    cc = getindice((boxwidth/ 2.0), dx, pmllayers[0])
    # vline = [(eyn(cc, i), dy[i]) for i in range(getindice(boxheight, dy, pmllayers[2]))]
    return (dx, dy, er, erxx, eryy, erzz, metalnodesx, metalnodesy, metalnodesz, cloops, vlines, parts)

def plot_parts(ekran, parts, solid=False, shift=(0,0)):
    """

    :param solid: if 1, use fill=true, else fill=false
    :type ekran: Matplotlib Axes object
    """
    # ekran.add_patch(Rectangle((0.0, 0.0), p[4], p[5], fc='r'))
    for p in parts:
        if p[1] == "rect":
            if p[0] == "dielectric":
                xo = p[2] - (p[4]/ 2.0)
                yo = p[3] - (p[5]/ 2.0)
                ekran.axes.add_patch(mpatches.Rectangle((xo+shift[0], yo+shift[1]), p[4], p[5], fill=solid, fc='green'))
            elif p[0] == "metal":
                xo = p[2] - (p[4]/ 2.0)
                yo = p[3] - (p[5]/ 2.0)
                ekran.axes.add_patch(mpatches.Rectangle((xo+shift[0], yo+shift[1]), p[4], p[5], fill=solid, fc='red'))


def plot_pml(ekran, pmllayers, dx, dy, solid=False, shift=(0,0)):
    """

    :param solid: if 1, use fill=true, else fill=false
    :type ekran: Matplotlib Axes object
    """
    # ekran.add_patch(Rectangle((0.0, 0.0), p[4], p[5], fc='r'))
    x1,x2,y1,y2=tuple(pmllayers)
    lx=np.sum(dx[:])
    ly=np.sum(dy[:])
    if x1>0:
        ekran.axes.add_patch(mpatches.Rectangle((0, 0), np.sum(dx[:x1]),ly, fill=solid, fc='blue'))
    if x2>0:
        ekran.axes.add_patch(mpatches.Rectangle((lx-np.sum(dx[-x2:]), 0), np.sum(dx[-x2:]),ly, fill=solid, fc='blue'))
    if y1>0:
        ekran.axes.add_patch(mpatches.Rectangle((0, 0), lx, np.sum(dy[:y1]), fill=solid, fc='blue'))
    if y2>0:
        ekran.axes.add_patch(mpatches.Rectangle((0, ly-np.sum(dy[-y2:]), 0), lx, np.sum(dy[-y2:]), fill=solid, fc='blue'))


def plot_grids(ekran, dx, dy):
    boxwidth = sum(dx)
    boxheight = sum(dy)
    xp = deepcopy(dx)
    yp = deepcopy(dy)
    for i in range(1, len(dx)):
        xp[i] = xp[i] + xp[i - 1]
    for i in range(1, len(dy)):
        yp[i] = yp[i] + yp[i - 1]
    xp = [0.0] + xp
    yp = [0.0] + yp
    for xo in xp:
        ekran.axes.plot([xo, xo], [0.0, boxheight], "black")
    for yo in yp:
        ekran.axes.plot([0.0, boxwidth], [yo, yo], "black")


def plot_fields(dx, dy, eigs, parts):
    boxwidth = sum(dx)
    boxheight = sum(dy)
    xi = np.linspace(0, boxwidth, 100)
    yi = np.linspace(0, boxheight, 100)
    for la, v in eigs:
        xp = deepcopy(dx)
        yp = deepcopy(dy)
        for i in range(1, len(dx)):
            xp[i] = xp[i] + xp[i - 1]
        for i in range(1, len(dy)):
            yp[i] = yp[i] + yp[i - 1]
        xtemp = []
        ytemp = []
        Ey = []
        for j in range(len(dy)):
            xtemp += [0.0, boxwidth]
            ytemp += [yp[j] - (dy[j]/ 2.0), yp[j] - (dy[j]/ 2.0)]
            Ey += [0.0, 0.0]
            for i in range(1, len(dx)):
                xtemp.append(xp[i - 1])
                ytemp.append(yp[j] - (dy[j]/ 2.0))
                Ey.append(v[eyn(i, j)])
        Eyy = griddata((xtemp, ytemp), np.real(Ey), (xi[None, :], yi[:, None]), method='cubic')
        xtemp = []
        ytemp = []
        Ex = []
        for i in range(len(dx)):
            xtemp += [xp[i] - (dx[i]/ 2.0), xp[i] - (dx[i]/ 2.0)]
            ytemp += [0.0, boxheight]
            Ex += [0.0, 0.0]
            for j in range(1, len(dy)):
                xtemp.append(xp[i] - (dx[i]/ 2.0))
                ytemp.append(yp[j - 1])
                Ex.append(v[exn(i, j)])
        Exx = griddata((xtemp, ytemp), np.real(Ex), (xi[None, :], yi[:, None]), method='cubic')
        import matplotlib.colorbar as mcolorbar
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        import matplotlib.patches as patches
        nz = mcolors.Normalize()
        mag = np.sqrt(Exx * Exx + Eyy * Eyy)
        nz.autoscale(mag)
        streamplot(xi, yi, Exx, Eyy, color=mag, density=3.0, cmap=cm.gist_rainbow, linewidth=1, arrowstyle='->')
        colorbar()
        # gca().add_patch(patches.Rectangle((boxwidth / 2.0 - metalwidth / 2.0, dielheight), metalwidth, metalthickness))
    legend(bbox_to_anchor=[1.0, 1.0])
    show()


def plot_field(ekran, dx, dy, v, parts, pmllayers, surface=True, comp="Ex"):
    """

    :type comp: "Ex","Ey"...
    :type surface: if 1, use contourf, else streamplot
    """
    # x1, x2, y1, y2 = tuple(pmllayers)
    # dx = [dx[0]] * x1 + dx + [dx[-1]] * x2
    # dy = [dy[0]] * y1 + dy + [dy[-1]] * y2

    # Nx = len(dx)
    # Ny = len(dy)
    # exn.Nx = eyn.Nx = hxn.Nx = hyn.Nx = Nx
    # exn.Ny = eyn.Ny = hxn.Ny = hyn.Ny = Ny

    print("ldy ",len(dy), len(dx))
    print("comp= ",comp)
    print("surface= ",surface)
    boxwidth = sum(dx)
    boxheight = sum(dy)
    xi = np.linspace(0, boxwidth, 100)
    yi = np.linspace(0, boxheight, 100)
    xp = deepcopy(dx)
    yp = deepcopy(dy)
    for i in range(1, len(dx)):
        xp[i] = xp[i] + xp[i - 1]
    for i in range(1, len(dy)):
        yp[i] = yp[i] + yp[i - 1]
    xtemp = []
    ytemp = []
    Ey = []
    Hx=[]
    for j in range(len(dy)):
        xtemp += [0.0, boxwidth]
        ytemp += [yp[j] - (dy[j]/ 2.0), yp[j] - (dy[j]/ 2.0)]
        Ey += [0.0, 0.0]
        Hx += [0.0, 0.0]
        for i in range(1, len(dx)):
            xtemp.append(xp[i - 1])
            ytemp.append(yp[j] - (dy[j]/ 2.0))
            Ey.append(v[0][eyn(i, j)])
            Hx.append(v[1][hxn(i, j)])
    Eyy = griddata((xtemp, ytemp), Ey, (xi[None, :], yi[:, None]), method='cubic')
    Hxx = griddata((xtemp, ytemp), Hx, (xi[None, :], yi[:, None]), method='cubic')
    xtemp = []
    ytemp = []
    Ex = []
    Hy = []
    TEx=[]
    for i in range(len(dx)):
        xtemp += [xp[i] - (dx[i]/ 2.0), xp[i] - (dx[i]/ 2.0)]
        ytemp += [0.0, boxheight]
        Ex += [0.0, 0.0]
        Hy += [0.0, 0.0]
        tt = []
        for j in range(1, len(dy)):
            xtemp.append(xp[i] - (dx[i]/ 2.0))
            ytemp.append(yp[j - 1])
            Ex.append(v[0][exn(i, j)])
            tt.append(v[0][exn(i, j)])
            Hy.append(v[1][hyn(i, j)])
        TEx.append(tt)
    # plt.imshow(np.abs(np.array(TEx)))
    # plt.show()
    # np.savetxt("TEx.txt",np.array(TEx))
    Exx = griddata((xtemp, ytemp), Ex, (xi[None, :], yi[:, None]), method='cubic')
    Hyy = griddata((xtemp, ytemp), Hy, (xi[None, :], yi[:, None]), method='cubic')
    magE = np.sqrt(np.abs(Exx * Exx + Eyy * Eyy))
    magH = np.sqrt(np.abs(Hxx * Hxx + Hyy * Hyy))

    # ekran.axes.streamplot(xi, yi, np.real(Exx), np.real(Eyy), color=mag, density=10.0, cmap=cm.gist_rainbow, linewidth=1, arrowstyle='->')

    if not hasattr(plot_field, "cb"):
        plot_field.cb=None

    if surface:
        Quantity=[]
        if comp == "Ex":
            Quantity = np.abs(Exx)
            nz.autoscale(magE)
        elif comp == "Ey":
            Quantity = np.abs(Eyy)
            nz.autoscale(magE)
        elif comp == "E_t":
            Quantity = np.abs(magE)
            nz.autoscale(magE)
        elif comp == "Hx":
            Quantity = np.abs(Hxx)
            nz.autoscale(magH)
        elif comp == "Hy":
            Quantity = np.abs(Hyy)
            nz.autoscale(magH)
        elif comp == "H_t":
            Quantity = np.abs(magH)
            nz.autoscale(magH)
        X, Y = np.meshgrid(xi, yi)
        #ekran.figure.gca().cla()
        temp = ekran.axes.contourf(X, Y, Quantity, color='black', density=10.0)
        if plot_field.cb:
            plot_field.cb.ax.clear()
            plot_field.cb = ekran.figure.colorbar(temp, cax=plot_field.cb.ax)
        else:

            plot_field.cb = ekran.figure.colorbar(temp)
    else:
        if comp=="E_t":
            nz.autoscale(magE)
            temp = ekran.axes.streamplot(xi, yi, np.real(Exx), np.real(Eyy), color=magE, density=10.0, cmap=cm.gist_rainbow,
                              linewidth=1, arrowstyle='->')
        elif comp == "H_t":
            nz.autoscale(magH)
            temp = ekran.axes.streamplot(xi, yi, np.real(Hxx), np.real(Hyy), color=magH, density=10.0, cmap=cm.gist_rainbow,
                                      linewidth=1, arrowstyle='->')    # ekran.axes.colorbar()
        if plot_field.cb:
            plot_field.cb.ax.clear()
            plot_field.cb = ekran.figure.colorbar(temp, cax=plot_field.cb.ax)
        else:
            plot_field.cb = ekran.figure.colorbar(temp)


if __name__ == "__main__":
    boxwidth = 0.03  # WG genişliği
    boxheight = 0.03 # WG yüksekliği
    dielheight = 0.014
    metalwidth = 0.01
    metalthickness = 0.002
    freq = 0.1e9
    epsilon = 5.0
    pml_layers=[0,0,0,0]
    # ms = microstrip(boxwidth, boxheight, dielheight, metalwidth, metalthickness, epsilon, freq)
    ms = multilayer(boxwidth, boxheight, [dielheight], [epsilon], [[dielheight, metalwidth, metalthickness,0.0]], freq, pml_layers,0.001,0.002)
    #ms = waveguide(boxwidth, boxheight, [dielheight], dielheight, metalwidth, metalthickness, [epsilon], freq, pml_layers,0.001,0.001)
    fig = plt.figure()
    out = fdfd_solve(*(list(ms[:-1]) + [freq]+[pml_layers]))
    plot_field(fig.gca(), ms[0], ms[1], out[0][-1], ms[-1],pml_layers, surface=True,comp="E_t")
    # plot_field(fig.gca(), ms[0], ms[1], out[0][-1], ms[-1],pml_layers, surface=True,comp="H_t")
    plot_grids(fig.gca(), ms[0], ms[1])
    plot_parts(fig.gca(), ms[-1])
    print(out[0][0],out[0][1])

    # print "eigs "
    # for i in out:
    #     print "i= ", i[0], i[1]
    # print ms[-2]

    # exn.Nx = eyn.Nx = hxn.Nx = hyn.Nx = Nx = 3
    # exn.Ny = eyn.Ny = hxn.Ny = hyn.Ny = Ny = 2
    # ind={}
    # for i in range(Nx+1):
        # for j in range(Ny+1):
            # print "exnnn ",i,j,exn.Nx,exn.Ny
            # if exn(i,j) is not None::
                # print "exn ",i,j,exn(i,j)
    # for i in range(Nx+1):
        # for j in range(Ny+1):
            # if eyn(i,j) is not None::
                # print "eyn ",i,j,eyn(i,j)
    # for i in range(Nx+1):
        # for j in range(Ny+1):
            # if hxn(i,j) is not None::
                # print "hxn ",i,j,hxn(i,j)
    # for i in range(Nx+1):
        # for j in range(Ny+1):
            # if hyn(i,j) is not None::
                # print "hyn ",i,j,hyn(i,j)

    plt.show()
