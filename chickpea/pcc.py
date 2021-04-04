# %%

import numpy as np
import legume

# %%

# The idea is to create a class that automatically defines cavity geometries for you. Starting with the only changes to the cavity being shifts.
# All of our crystals, including the shifts, have mirror symmetry about the x and y axes.

'''
inputs:
    crystal: A string that's either 'H' or 'L', depending on what the desired cavity type is
    rads: an array of radii of the holes in the pcc.
        this array is internally cast to one whose size is the number of holes in the first quadrant.
        if the length of the given array is less than this, then the radii of the first 
    supercell: an array of length 2, defining the supercell of the pcc.
    thickness: the thickness of the pcc layer
    eps: background permittivity of the pcc
    m: first cavity parameter. For instance, for an L3 crystal, this would be 3.
    n: second cavity parameter. For instance, for an L3-4 crystal, this would be 4.
    *displacements: optional displacements argument.
    **kwargs: maybe add ambient radius?

methods:
    cavity:
        inputs: nothing
        outputs: a crystal object based on the initialization parameters
    getsupercell:
        returns the supercell
    getnumholes:
        returns the number of holes in the first quadrant of the given crystal.
    getbase:
        returns the base crystal, without displacements or the cavity.
'''


class pcc:
    #dx and dy should be optional arguments
    def __init__(self, crystal, rads, supercell, thickness, eps, m, n, *displacements, **kwargs):
        #not sure if i have to store all these once i create the appropriate lattice...
        self.crystal = [crystal, m, n]
        self.supercell = supercell
        self.thick = thickness
        self.eps = eps
        lattice = legume.Lattice([supercell[0], 0], [0, supercell[1]*np.sqrt(3)/2])
        self.lattice = lattice
        if crystal == 'L':
            self.numholes = (supercell[0]//2+1)*(supercell[1]//2+1) - (m+1)//2 + (n+1)//2
        else:
            numholes = (supercell[0]//2+1)*(supercell[1]//2+1)
            for i in range(m):
                numholes = numholes - (m+1+i)//2
            self.numholes = numholes
        if len(rads) > self.numholes:
            print("too many radii defined. taking the first " + str(self.numholes)+ " radii.")
            self.rads = rads[:self.numholes]
        elif len(rads) == self.numholes:
            self.rads = rads
        elif len(rads) < self.numholes:
            radarray = list(np.sqrt(np.sum(np.square(rads))/len(rads))*np.ones(self.numholes))
            radarray[:len(rads)] = rads
            self.rads = radarray
        #gotta deal with displacements properly.
        if len(displacements) == 0:
            self.dx = np.zeros(self.numholes)
            self.dy = np.zeros(self.numholes)
        #if only one displacement array, then assume it's displacements in x direction.
        #this assumption could (and probably should) be changed based on the type of crystal
        elif len(displacements) == 1:
            self.dy = np.zeros(self.numholes)
            if len(displacements[0]) > self.numholes:
                print("too many displacements defined. taking the first " + str(self.numholes)+ " displacements.")
                self.dx = displacements[0][:self.numholes]
            elif len(displacements[0]) == self.numholes:
                self.dx = displacements[0]
            elif len(displacements[0]) < self.numholes:
                dxarray = np.zeros(self.numholes)
                dxarray[:len(displacements[0])] = displacements[0]
                self.dx = dxarray
        elif len(displacements) == 2:
            if len(displacements[0]) > self.numholes:
                print("too many displacements defined. taking the first " + str(self.numholes)+ " displacements.")
                self.dx = displacements[0][:self.numholes]
            elif len(displacements[0]) == self.numholes:
                self.dx = displacements[0]
            elif len(displacements[0]) < self.numholes:
                dxarray = np.zeros(self.numholes)
                dxarray[:len(displacements[0])] = displacements[0]
                self.dx = dxarray
            if len(displacements[1]) > self.numholes:
                print("too many displacements defined. taking the first " + str(self.numholes)+ " displacements.")
                self.dy = displacements[1][:self.numholes]
            elif len(displacements[1]) == self.numholes:
                self.dy = displacements[1]
            elif len(displacements[1]) < self.numholes:
                dyarray = np.zeros(self.numholes)
                dyarray[:len(displacements[1])] = displacements[1]
                self.dy = dyarray
        else:
            print("displacement array too large.")
        #FIGURE OUT HOW TO THROW EXCEPTIONS / DEAL WITH BAD INPUTS

    def cavity(self):
        m, n = self.crystal[1:]
        ctype = self.crystal[0]
        #generate holes. Note H0 cavity is different wrt. this.
        Nx, Ny = self.supercell
        xp, yp = [], []
        nx, ny = Nx//2 + 1, Ny//2 + 1

        if m==0 or (ctype == 'L' and m%2==0):
            for iy in range(ny):
                for ix in range(nx):
                    xp.append(ix + ((iy+1)%2)*0.5)
                    yp.append(iy*np.sqrt(3)/2)
        else:
            for iy in range(ny):
                for ix in range(nx):
                    xp.append(ix + (iy%2)*0.5)
                    yp.append(iy*np.sqrt(3)/2)

        def removeholes(xp, yp, Nx, Ny, ctype, m, n):
            #check if crystal valid
            if (m+1)//2 >= min((Nx+1)//2, (Ny+1)//2):
                print("cavity invalid - use a bigger supercell")
                return xp, yp
            
            #not sure if necessary
            if m == 0:
                return xp, yp

            elif ctype == 'L':
                #remove m holes:
                xremoved = xp.copy()[(m+1)//2:]
                yremoved = yp.copy()[(m+1)//2:]
                
                #fill with n holes:
                xfill = list(np.linspace(-xremoved[0], xremoved[0], num = n+2, endpoint = True)[(n+2)//2: -1])
                yfill = list(np.zeros((n+1)//2))
                
                xnew = xfill + xremoved
                ynew = yfill + yremoved
                return xnew, ynew
            
            elif ctype == 'H':
                xnew = xp.copy()
                ynew = yp.copy()
                
                for i in range(m):
                    ind = (m-1-i)*(Nx//2)+m-i-1
                    num = (m+1+i)//2
                    del xnew[ind:ind+num]
                    del ynew[ind:ind+num]
                return xnew, ynew
                
            else:
                print("invalid crystal type")
                return xp, yp

        xp, yp = removeholes(xp, yp, Nx, Ny, ctype, m, n)
        self.xp, self.yp = xp, yp #not sure if this should be here now, since xp and yp aren't initialized in the constructor
        cryst = legume.PhotCryst(self.lattice)
        cryst.add_layer(d=self.thick, eps_b=self.eps)

        for ic, x in enumerate(xp):
            yc = yp[ic] if yp[ic] == 0 else yp[ic] + self.dy[ic]
            xc = x if x == 0 else xp[ic] + self.dx[ic]
            cryst.add_shape(legume.Circle(x_cent=xc, y_cent=yc, r=self.rads[ic]))
        
            if nx-0.6 > xp[ic] > 0 and (ny-1.1)*np.sqrt(3)/2 > yp[ic] > 0:
                cryst.add_shape(legume.Circle(x_cent=-xc, y_cent=-yc, r=self.rads[ic]))
            if nx-1.6 > xp[ic] > 0:
                cryst.add_shape(legume.Circle(x_cent=-xc, y_cent=yc, r=self.rads[ic]))
            if (ny-1.1)*np.sqrt(3)/2 > yp[ic] > 0 and nx-1.1 > xp[ic]:
                cryst.add_shape(legume.Circle(x_cent=xc, y_cent=-yc, r=self.rads[ic]))

        #et voila! the crystal should be defined.
        return cryst

    def getsupercell(self):
        return self.supercell.copy()
    
    def getnumholes(self):
        return self.numholes

    def getbase(self):
        lattice = legume.Lattice('hexagonal')
        avgrad = np.sqrt(np.sum(np.square(self.rads))/len(self.rads))
        cryst = legume.PhotCryst(lattice)
        cryst.add_layer(d=self.thick, eps_b=self.eps)
        cryst.add_shape(legume.Circle(x_cent = 0, y_cent = 0, r=avgrad))
        return cryst


