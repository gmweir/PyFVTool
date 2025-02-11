"""
Boundary condition classes


Notes
-----

"""

import numpy as np
from .versioning import *
from .mesh import *
from .utilities import *


class Boundary:
    def __init__(self, a: np.ndarray, b: np.ndarray, c: np.ndarray, periodic=False):
        self.a = a
        self.b = b
        self.c = c
        self.periodic = periodic

    def __str__(self):
        temp = vars(self)
        for item in temp:
            print(item, ':', temp[item])
        return ""

    def __repr__(self):
        temp = vars(self)
        for item in temp:
            print(item, ':', temp[item])
        return ""


class BoundaryCondition:
    def __init__(self, mesh: MeshStructure,
                 left: Boundary, right: Boundary,
                 bottom: Boundary, top: Boundary,
                 back: Boundary, front: Boundary):
        self.domain = mesh
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.back = back
        self.front = front

    def __str__(self):
        temp = vars(self)
        for item in temp:
            print(item, ':', temp[item])
        return ""

    def __repr__(self):
        temp = vars(self)
        for item in temp:
            print(item, ':', temp[item])
        return ""


class BoundaryCondition1D(BoundaryCondition):
    def __init__(self, mesh: Mesh1D):
        left = Boundary(np.array([1.0]), np.array([0.0]), np.array([0.0]))
        right = Boundary(np.array([1.0]), np.array([0.0]), np.array([0.0]))
        bottom = Boundary(np.array([]), np.array([]), np.array([]))
        top = Boundary(np.array([]), np.array([]), np.array([]))
        back = Boundary(np.array([]), np.array([]), np.array([]))
        front = Boundary(np.array([]), np.array([]), np.array([]))
        BoundaryCondition.__init__(
            self, mesh, left, right, bottom, top, back, front)


class BoundaryCondition2D(BoundaryCondition):
    def __init__(self, mesh: Mesh2D):
        Nx, Ny = mesh.dims
        left = Boundary(np.ones(Ny), np.zeros(Ny), np.zeros((1, Ny)))
        right = Boundary(np.ones(Ny), np.zeros(Ny), np.zeros(Ny))
        bottom = Boundary(np.ones(Nx), np.zeros(Nx), np.zeros(Nx))
        top = Boundary(np.ones(Nx), np.zeros(Nx), np.zeros(Nx))
        back = Boundary(np.array([]), np.array([]), np.array([]))
        front = Boundary(np.array([]), np.array([]), np.array([]))
        BoundaryCondition.__init__(self, mesh, left, right, bottom, top, back, front)


class BoundaryCondition3D(BoundaryCondition):
    def __init__(self, mesh: Mesh3D):
        Nx, Ny, Nz = mesh.dims
        left = Boundary(np.ones((Ny, Nz)), np.zeros((Ny, Nz)), np.zeros((Ny, Nz)))
        right = Boundary(np.ones((Ny, Nz)), np.zeros((Ny, Nz)), np.zeros((Ny, Nz)))
        bottom = Boundary(np.ones((Nx, Nz)), np.zeros((Nx, Nz)), np.zeros((Nx, Nz)))
        top = Boundary(np.ones((Nx, Nz)), np.zeros((Nx, Nz)), np.zeros((Nx, Nz)))
        back = Boundary(np.ones((Nx, Ny)), np.zeros((Nx, Ny)), np.zeros((Nx, Ny)))
        front = Boundary(np.ones((Nx, Ny)), np.zeros((Nx, Ny)), np.zeros((Nx, Ny)))
        BoundaryCondition.__init__(self, mesh, left, right, bottom, top, back, front)


def createBC(mesh: MeshStructure):
    if issubclass(type(mesh), Mesh1D):
        return BoundaryCondition1D(mesh)
    elif issubclass(type(mesh), Mesh2D):
        return BoundaryCondition2D(mesh)
    elif issubclass(type(mesh), Mesh3D):
        return BoundaryCondition3D(mesh)

    
def BC2GhostCells(phi):   
    """
    Assign values at teh bounadries to the ghost cells
    
    Parameters
    ----------
    phi : {CellVariable object}
        Variable defined at mesh-nodes
        
    Returns
    -------
    out : {CellVariable object}
        Updated variable defined at mesh-nodes with ghost cells
        

    See Also
    --------

    Notes
    -----
    phi is a CellVariable, but specifying it here would create a circular reference, as this module is imported into the cell-module

    Examples
    --------

    """
    d = phi.domain.dimension
    phi = phi.copy()
    if (d == 1 ) or (d==1.5) or (d==1.8):
        phi.value[0] = 0.5*(phi.value[0]+phi.value[1])
        phi.value[-1] = 0.5*(phi.value[-1]+phi.value[-2])
        
    elif (d == 2) or (d == 2.5) or (d == 2.8):
        phi.value[0, 1:-2] = 0.5*(phi.value[0, 1:-2] + phi.value[1, 1:-2])  # (1,2:end-1)
        phi.value[-1, 1:-2] = 0.5*(phi.value[-1, 1:-2] + phi.value[-2, 1:-2])    # (end,2:end-1)
        phi.value[1:-2, 0] = 0.5*(phi.value[1:-2, 0] + phi.value[1:-2, 1])         # (2:end-1, 1)
        phi.value[1:-2, -1] = 0.5*(phi.value[1:-2, -1] + phi.value[1:-2, -2])  # (2:end-1, end) 
        
    elif (d == 3) or (d == 3.2):
        phi.value[0, 1:-2, 1:-2] = 0.5*(phi.value[0, 1:-2, 1:-2] + phi.value[1, 1:-2, 1:-2])  # (1, 2:end-1, 2:end-1)
        phi.value[-1, 1:-2, 1:-2] = 0.5*(phi.value[-1, 1:-2, 1:-2] + phi.value[-2, 1:-2, 1:-2]) # (end, 2:end-1, 2:end-1)
        phi.value[1:-2, 0, 1:-2] = 0.5*(phi.value[1:-2, 0, 1:-2] + phi.value[1:-2, 1, 1:-2]) # (2:end-1, 1, 2:end-1)
        phi.value[1:-2, -1, 1:-2] = 0.5*(phi.value[1:-2, -1, 1:-2] + phi.value[1:-2, -2, 1:-2])  # (2:end-1, end, 2:end-1)
        phi.value[1:-2, 1:-2, 0] = 0.5*(phi.value[1:-2, 1:-2, 0] + phi.value[1:-2, 1:-2, 1])     # (2:end-1, 2:end-1, 1)
        phi.value[1:-2, 1:-2, -1] = 0.5*(phi.value[1:-2, 1:-2, -2])    # (2:end-1, 2:end-1, end)
    return phi


def combineBC(BC: BoundaryCondition, Meq: csr_array, RHSeq: csr_array):
    """
    Combine boundary conditions to return a modified coefficient matrix and RHS vector.
    
    This function combines the boundary condition equatiosn with the new main physical model equations (Meq), and delivers the matrix of coefficients and right-hand-side of the equation to be solved for the internal cells.  It is useful if one needs to use an ODE solver for the accumulation term, i.e., 
        d(phi)/dt = M phi
        
    Parameters
    ----------
    BC : {BoundaryCondition object}
        Class object structure with boundary condtion definitions
        
    Meq : {csr_array}
        Coefficient matrix defining equations to be solved
        
    RHSeq : {csr_array}
        Coefficient matrix defining right-hand side of the equations

    Returns
    -------
    Mout : {csr_array}
        Altered coefficient matrix including boundary condition terms
        
    RHSout : {csr_array}
        Altered right-hand-side coefficient vector including boundary condition terms

    See Also
    --------
    combineBC1D : combine boundary conditions for a 1D grid
    combineBC2D : combine boundary conditions for a 2D grid
    combineBC3D : combine boundary conditions for a 3D grid

    Notes
    -----

    Examples
    --------
    >>>     
    
    
    """
    # Number of dimensions on domain (and therefore boundary conditions)
    d = BC.domain.dimension
    
    if (d == 1) or (d == 1.5):
        return combineBC1D(BC, Meq, RHSeq)
    elif (d == 2) or (d == 2.5):
        return combineBC2D(BC, Meq, RHSeq)
    elif (d == 3):
        return combineBC3D(BC, Meq, RHSeq)
    
    
def combineBC1D(BC: BoundaryCondition, Meq: csr_array, RHSeq: csr_array):
    """
    Wrap boundary conditions into the problem coefficient matrix and RHS-vector
    
    -->  Errors abound here
    
    """
    # extract data from the mesh structure
    Nx = BC.domain.dims[0]
    dx_1 = BC.domain.cellsize.x[0]
    dx_end= BC.domain.cellsize.x[-1]
    G = np.asarray(range(Nx+2))   # G = 1:Nx+2;

    # define the RHS column vector
    ms = np.shape(Meq)
    
    # Copy input matrices
    M = Meq
    RHS = RHSeq

    # Assign values to the boundary condition matrix and the RHS vector based
    # on the BC structure

    # Right boundary
    ii = Nx+1  # Nx+2
    right = sub2ind(ms, G[ii-1], G[ii-1])  # right boundary cells
    rightE = sub2ind(ms, G[ii-1], G[ii])   # east cells to right boundary cells
    
    try:
        print(M[right])
        print(M[rightE])

        M[right] = M[right]-((BC.right.b/2.0 - BC.right.a/dx_end)/(BC.right.b/2.0 + BC.right.a/dx_end))*M[rightE]    
        RHS[G[ii-1]] = RHS[G[ii-1]]-M[rightE]*BC.right.c/(BC.right.b/2.0 + BC.right.a/dx_end)
    except:
        print((M.shape, right, rightE))
        print((RHS.shape, G[ii-1]))
        raise
        
    # Left boundary
    ii = 0
    left = sub2ind(ms, G[ii+1], G[ii+1])  # left boundary cells
    leftW = sub2ind(ms, G[ii+1], G[ii])   # west cells to left boundary cells
    M[left] = M[left]-((BC.left.b/2.0 + BC.left.a/dx_1)/(BC.left.b/2.0 - BC.left.a/dx_1))*M[leftW]
    RHS[G[ii+1]] = RHS[G[ii+1]]-M[leftW]*BC.left.c/(BC.left.b/2.0 - BC.left.a/dx_1)

    return M[G[1:-1], G[1:-1]], RHS[G[1:-1].reshape(Nx, 1, order='F')]


def combineBC2D(BC: BoundaryCondition, Meq: csr_array, RHSeq: csr_array):
    """   
    
    """
    return NotImplemented

def combineBC3D(BC: BoundaryCondition, Meq: csr_array, RHSeq: csr_array):
    """   
    
    """
    return NotImplemented


"""
Discretizing boundary conditions to csr array
Example:
row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
csr_array((data, (row, col)), shape=(3, 3)).toarray()
array([[1, 0, 2],
       [0, 0, 3],
       [4, 5, 6]])
"""


def boundaryCondition(BC: BoundaryCondition):
    """
    Creates and returns the matrix of coefficients (equations) and the RHS-vector.
    
    Calculates the coefficient matrix for the system of equations, and also returns the right-hand-side vector, including boundary conditions as effective source terms.
        function [BCMatrix, BCRHS] = boundaryCondition(BC)

    Parameters
    ----------
    BC : {Boundary Condition object}
        Container for boundary condition data.  
        
    Returns
    -------
    BCMatrix : {np.ndarray}
        A square sparse matrix - matrix of coefficients for the given boundary conditions (not including system of equations)
        
    BCRHS : {np.ndarray}
        A column of vecotrs - vector of effective source terms caused by the the given boundary conditions (not including real source terms)

    See Also
    --------
    createBC : 
    createMesh1D :
    createMesh2D :
    createMesh3D :
    createMeshCylindrical1D :
    createMeshCylindrical2D :
    createMeshCylindrical3D :    NotImplemented
    createMeshRadial2D :
    createMeshCylindrical3D :
    cellBoundary :
    combineBC :
    createCellVariable :

    Notes
    -----

    Examples
    --------
    >>> L = 1.0  # Length of a 1D domain
    L = 1.0
    >>> Nx = 5   # Number of grids in x-direction
    Nx = 5
    >>> m = createMesh1D(Nx, L)
    m
    >>> BC = createBC(m)  # all Neumann BC's
    BC
    >>> Mbc, RHSbc = boundaryCondition(BC)
    ...
    >>> print(Mbc)
    ...


    """
    d = BC.domain.dimension
    if (d ==1) or (d==1.5) or (d==1.8):
        BCMatrix, BCRHS = boundaryCondition1D(BC)
    elif (d == 2) or (d == 2.5):
        BCMatrix, BCRHS = boundaryCondition2D(BC)
    elif (d == 2.8):
        BCMatrix, BCRHS = boundaryConditionRadial2D(BC)
    elif (d == 3):
        BCMatrix, BCRHS = boundaryCondition3D(BC)
    elif (d == 3.2):
        BCMatrix, BCRHS = boundaryConditionCylindrical3D(BC)   
    return BCMatrix, BCRHS


def boundaryCondition1D(BC: BoundaryCondition1D):
    Nx = BC.domain.dims[0]
    dx_1 = BC.domain.cellsize.x[0]
    dx_end = BC.domain.cellsize.x[-1]
    G = int_range(1, Nx+2)-1

    nb = 8  # number of boundary nodes
    ii = np.zeros(nb, dtype=int)
    jj = np.zeros(nb, dtype=int)
    s = np.zeros(nb, dtype=float)

    BCRHS = np.zeros(Nx+2)  # RHS vector
    q = -1  # a counter in case I'm wrong with my count of nonzero elements

    if (not BC.left.periodic) and (not BC.right.periodic):
        # right boundary
        i = Nx+1  # -1 for python 0-based indexing
        q += 1
        ii[q] = G[i]
        jj[q] = G[i]
        s[q] = BC.right.b/2 + BC.right.a/dx_end
        q = q+1
        ii[q] = G[i]
        jj[q] = G[i-1]
        s[q] = BC.right.b/2 - BC.right.a/dx_end
        BCRHS[G[i]] = BC.right.c

        # Left boundary
        i = 0
        q += 1
        ii[q] = G[i]
        jj[q] = G[i+1]
        s[q] = -(BC.left.b/2 + BC.left.a/dx_1)
        q = q+1
        ii[q] = G[i]
        jj[q] = G[i]
        s[q] = -(BC.left.b/2 - BC.left.a/dx_1)
        BCRHS[G[i]] = -BC.left.c
    elif BC.right.periodic or BC.left.periodic:  # periodic boundary condition
        # Right boundary
        i = Nx+1
        q = q+1
        ii[q] = G[i]
        jj[q] = G[i]
        s[q] = 1
        q = q+1
        ii[q] = G[i]
        jj[q] = G[i-1]
        s[q] = -1
        q = q+1
        ii[q] = G[i]
        jj[q] = G[0]
        s[q] = dx_end/dx_1
        q = q+1
        ii[q] = G[i]
        jj[q] = G[1]
        s[q] = -dx_end/dx_1
        BCRHS[G[i]] = 0
        # Left boundary
        i = 0
        q = q+1
        ii[q] = G[i]
        jj[q] = G[i]
        s[q] = 1.0
        q = q+1
        ii[q] = G[i]
        jj[q] = G[1]
        s[q] = 1.0
        q = q+1
        ii[q] = G[i]
        jj[q] = G[Nx]
        s[q] = -1.0
        q = q+1
        ii[q] = G[i]
        jj[q] = G[Nx+1]
        s[q] = -1.0
        BCRHS[G[i]] = 0
    # Build the sparse matrix of the boundary conditions
    q += 1
    BCMatrix = csr_array((s[0:q], (ii[0:q], jj[0:q])), shape=(Nx+2, Nx+2))
    return BCMatrix, BCRHS


def boundaryCondition2D(BC: BoundaryCondition2D):
    Nx, Ny = BC.domain.dims
    dx_1 = BC.domain.cellsize.x[0]
    dx_end = BC.domain.cellsize.x[-1]
    dy_1 = BC.domain.cellsize.y[0]
    dy_end = BC.domain.cellsize.y[-1]
    G = BC.domain.cell_numbers()

    nb = 8*(Nx+Ny+2)  # number of boundary nodes
    ii = np.zeros(nb, dtype=int)
    jj = np.zeros(nb, dtype=int)
    s = np.zeros(nb, dtype=float)

    BCRHS = np.zeros((Nx+2)*(Ny+2))  # RHS vector
    q = -1  # a counter in case I'm wrong with my count of nonzero elements
    #  assign value to the corner nodes (useless cells)
    q = int_range(0, 3)
    ii[q] = BC.domain.corners
    jj[q] = BC.domain.corners
    s[q] = np.max(BC.top.b/2 + BC.top.a/dy_end)
    BCRHS[BC.domain.corners] = 0.0

    if (not BC.top.periodic) and (not BC.bottom.periodic):
        # top boundary
        j=Ny+1
        i=int_range(1,Nx)
        q = q[-1]+i
        ii[q] = G[i,j]  
        jj[q] = G[i,j]  
        s[q] = (BC.top.b/2 + BC.top.a/dy_end).ravel()
        q = q[-1]+i
        ii[q] = G[i,j]  
        jj[q] = G[i,j-1] 
        s[q] = (BC.top.b/2 - BC.top.a/dy_end).ravel()
        BCRHS[G[i,j]] = (BC.top.c).ravel()

        # Bottom boundary
        j=0
        # i=1:Nx already defined
        q = q[-1]+i
        ii[q] = G[i,j]  
        jj[q] = G[i,j+1]  
        s[q] = -(BC.bottom.b/2 + BC.bottom.a/dy_1).ravel()
        q = q[-1]+i
        ii[q] = G[i,j]  
        jj[q] = G[i,j] 
        s[q] = -(BC.bottom.b/2 - BC.bottom.a/dy_1).ravel()
        BCRHS[G[i,j]] = (-BC.bottom.c).ravel()
    elif BC.top.periodic or BC.bottom.periodic: # periodic boundary
        # top boundary
        j=Ny+1
        # i=int_range(1,Nx)
        i=int_range(1,Nx)
        q = q[-1]+i
        ii[q] = G[i,j]  
        jj[q] = G[i,j]  
        s[q] = 1
        q = q[-1]+i
        ii[q] = G[i,j]  
        jj[q] = G[i,j-1]  
        s[q] = -1
        q = q[-1]+i
        ii[q] = G[i,j]  
        jj[q] = G[i,0] 
        s[q] = dy_end/dy_1
        q = q[-1]+i
        ii[q] = G[i,j]  
        jj[q] = G[i,1] 
        s[q] = -dy_end/dy_1
        BCRHS[G[i,j]] = 0

        # Bottom boundary
        j=0
        # i=int_range(1,Nx)
        q = q[-1]+i
        ii[q] = G[i,j]  
        jj[q] = G[i,j]  
        s[q] = 1
        q = q[-1]+i
        ii[q] = G[i,j]  
        jj[q] = G[i,j+1]  
        s[q] = 1
        q = q[-1]+i
        ii[q] = G[i,j]
        jj[q] = G[i,Ny+1] 
        s[q] = -1
        q = q[-1]+i
        ii[q] = G[i,j]  
        jj[q] = G[i,Ny+2]
        s[q] = -1
        BCRHS[G[i,j]] = 0

    if (not BC.left.periodic) and (not BC.right.periodic):
        # right boundary
        i = Nx+1  # -1 for python 0-based indexing
        j = int_range(1, Ny)
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[i,j]
        s[q] = BC.right.b/2 + BC.right.a/dx_end
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[i-1,j]
        s[q] = BC.right.b/2 - BC.right.a/dx_end
        BCRHS[G[i,j]] = BC.right.c

        # Left boundary
        i = 0
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[i+1,j]
        s[q] = -(BC.left.b/2 + BC.left.a/dx_1)
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[i,j]
        s[q] = -(BC.left.b/2 - BC.left.a/dx_1)
        BCRHS[G[i,j]] = -BC.left.c
    elif BC.right.periodic or BC.left.periodic:  # periodic boundary condition
        # Right boundary
        i = Nx+1
        j = int_range(1, Ny)
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[i,j]
        s[q] = 1
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[i-1,j]
        s[q] = -1
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[0,j]
        s[q] = dx_end/dx_1
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[1,j]
        s[q] = -dx_end/dx_1
        BCRHS[G[i,j]] = 0
        # Left boundary
        i = 0
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[i,j]
        s[q] = 1.0
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[i+1,j]
        s[q] = 1.0
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[Nx,j]
        s[q] = -1.0
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[Nx+1,j]
        s[q] = -1.0
        BCRHS[G[i,j]] = 0.0
    # Build the sparse matrix of the boundary conditions
    q = q[-1] + 1
    BCMatrix = csr_array((s[0:q], (ii[0:q], jj[0:q])), 
                         shape=((Nx+2)*(Ny+2), (Nx+2)*(Ny+2)))
    return BCMatrix, BCRHS


def boundaryCondition3D(BC: BoundaryCondition3D):
    # extract data from the mesh structure
    Nx, Ny, Nz = BC.domain.dims
    G=BC.domain.cell_numbers()
    dx_1 = BC.domain.cellsize.x[0]
    dx_end = BC.domain.cellsize.x[-1]
    dy_1 = BC.domain.cellsize.y[0]
    dy_end = BC.domain.cellsize.y[-1]
    dz_1 = BC.domain.cellsize.z[0]
    dz_end = BC.domain.cellsize.z[-1]

    i_ind = int_range(1,Nx)[:, np.newaxis, np.newaxis]
    j_ind = int_range(1,Ny)[np.newaxis, :, np.newaxis]
    k_ind = int_range(1,Nz)[np.newaxis, np.newaxis, :]
    # number of boundary nodes (axact number is 2[(m+1)(n+1)*(n+1)*(p+1)+(m+1)*p+1]:
    nb = 8*((Nx+1)*(Ny+1)+(Nx+1)*(Nz+1)+(Ny+1)*(Nz+1))

    # define the vectors to be used for the creation of the sparse matrix
    ii = np.zeros(nb, dtype=int)
    jj = np.zeros(nb, dtype=int)
    s = np.zeros(nb, dtype=float)

    # define the RHS column vector
    BCRHS = np.zeros((Nx+2)*(Ny+2)*(Nz+2))

    # assign value to the corner nodes (useless cells)
    q = int_range(0, 7)
    ii[q] = BC.domain.corners 
    jj[q] = BC.domain.corners
    s[q] = 1.0
    BCRHS[BC.domain.corners] = 0.0

    # assign values to the edges (useless cells)
    q = q[-1]+int_range(1, np.size(BC.domain.edges))
    ii[q] = BC.domain.edges 
    jj[q] = BC.domain.edges
    s[q] = 1.0
    BCRHS[BC.domain.edges] = 0.0

    # Assign values to the boundary condition matrix and the RHS vector based
    # on the BC structure
    if (not BC.top.periodic) and (not BC.bottom.periodic):
        # top boundary
        j=Ny+1
        i = i_ind
        k = k_ind
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = (BC.top.b/2 + BC.top.a/dy_end).ravel()
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j-1,k].ravel()
        s[q] = (BC.top.b/2 - BC.top.a/dy_end).ravel()
        BCRHS[G[i,j,k].ravel()] = (BC.top.c).ravel()

        # Bottom boundary
        j=0
        i=i_ind
        k=k_ind
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j+1,k].ravel()
        s[q] = -(BC.bottom.b/2 + BC.bottom.a/dy_1).ravel()
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = -(BC.bottom.b/2 - BC.bottom.a/dy_1).ravel()
        BCRHS[G[i,j,k].ravel()] = -(BC.bottom.c).ravel()
    elif BC.top.periodic or BC.bottom.periodic: # periodic
        # top boundary
        j=Ny+1
        i=i_ind
        k=k_ind
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = 1.0
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j-1,k].ravel()
        s[q] = -1.0
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,0,k].ravel()
        s[q] = dy_end/dy_1
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,1,k].ravel()
        s[q] = -dy_end/dy_1
        BCRHS[G[i,j,k].ravel()] = 0.0

        # Bottom boundary
        j=0
        i=i_ind
        k=k_ind
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = 1.0
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j+1,k].ravel()
        s[q] = 1.0
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,Ny,k].ravel()
        s[q] = -1.0
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,Ny+1,k].ravel()
        s[q] = -1.0
        BCRHS[G[i,j,k].ravel()] = 0.0
    if (not BC.right.periodic) and (not BC.left.periodic):
        # Right boundary
        i=Nx+1
        j=j_ind
        k=k_ind
        q = q[-1]+int_range(1,Ny*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = (BC.right.b/2 + BC.right.a/dx_end).ravel()
        q = q[-1]+int_range(1,Ny*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i-1,j,k].ravel()
        s[q] = (BC.right.b/2 - BC.right.a/dx_end).ravel()
        BCRHS[G[i,j,k].ravel()] = (BC.right.c).ravel()

        # Left boundary
        i = 0
        j=j_ind
        k=k_ind
        q = q[-1]+int_range(1,Ny*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i+1,j,k].ravel()
        s[q] = -(BC.left.b/2 + BC.left.a/dx_1).ravel()
        q = q[-1]+int_range(1,Ny*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = -(BC.left.b/2 - BC.left.a/dx_1).ravel()
        BCRHS[G[i,j,k].ravel()] = -(BC.left.c).ravel()
    elif BC.right.periodic or BC.left.periodic: # periodic
        # Right boundary
        i=Nx+1
        j=j_ind
        k=k_ind
        q = q[-1]+int_range(1,Ny*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = 1.0
        q = q[-1]+int_range(1,Ny*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i-1,j,k].ravel()
        s[q] = -1.0
        q = q[-1]+int_range(1,Ny*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[0,j,k].ravel()
        s[q] = dx_end/dx_1
        q = q[-1]+int_range[1,Ny*Nz]
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[1,j,k].ravel()
        s[q] = -dx_end/dx_1
        BCRHS[G[i,j,k].ravel()] = 0.0

        # Left boundary
        i = 0
        j=j_ind
        k=k_ind
        q = q[-1]+int_range(1,Ny*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = 1.0
        q = q[-1]+int_range(1,Ny*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i+1,j,k].ravel()
        s[q] = 1.0
        q = q[-1]+int_range(1,Ny*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[Nx,j,k].ravel()
        s[q] = -1.0
        q = q[-1]+int_range(1,Ny*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[Nx+1,j,k].ravel()
        s[q] = -1.0
        BCRHS[G[i,j,k].ravel()] = 0.0
    if (not BC.front.periodic) and (not BC.back.periodic):
        # Front boundary
        k=Nz+1
        i = i_ind
        j = j_ind
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = (BC.front.b/2 + BC.front.a/dz_end).ravel()
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k-1].ravel()
        s[q] = (BC.front.b/2 - BC.front.a/dz_end).ravel()
        BCRHS[G[i,j,k].ravel()] = (BC.front.c).ravel()

        # Back boundary
        k=0
        i = i_ind
        j = j_ind
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k+1].ravel()
        s[q] = -(BC.back.b/2 + BC.back.a/dz_1).ravel()
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = -(BC.back.b/2 - BC.back.a/dz_1).ravel()
        BCRHS[G[i,j,k].ravel()] = -(BC.back.c).ravel()
    elif BC.front.periodic or BC.back.periodic: # periodic
        # Front boundary
        k=Nz+1
        i = i_ind
        j = j_ind
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = 1.0
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k-1].ravel()
        s[q] = -1.0
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,0].ravel()
        s[q] = dz_end/dz_1
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,1].ravel()
        s[q] = -dz_end/dz_1
        BCRHS[G[i,j,k].ravel()] = 0.0

        # Back boundary
        k=0
        i = i_ind
        j = j_ind
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = 1.0
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k+1].ravel()
        s[q] = 1.0
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,Nz].ravel()
        s[q] = -1.0
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,Nz+1].ravel()
        s[q] = -1.0
        BCRHS[G[i,j,k].ravel()] = 0.0

    # Build the sparse matrix of the boundary conditions
    q = q[-1]+1
    BCMatrix = csr_array((s[0:q], (ii[0:q], jj[0:q])), 
                         shape=((Nx+2)*(Ny+2)*(Nz+2), (Nx+2)*(Ny+2)*(Nz+2)))
    return BCMatrix, BCRHS


def boundaryConditionRadial2D(BC: BoundaryCondition2D):
    Nx, Ny = BC.domain.dims
    dx_1 = BC.domain.cellsize.x[0]
    dx_end = BC.domain.cellsize.x[-1]
    dy_1 = BC.domain.cellsize.y[0]
    dy_end = BC.domain.cellsize.y[-1]
    rp = BC.domain.cellcenters.x
    G = BC.domain.cell_numbers()

    nb = 8*(Nx+Ny+2)  # number of boundary nodes
    ii = np.zeros(nb, dtype=int)
    jj = np.zeros(nb, dtype=int)
    s = np.zeros(nb, dtype=float)

    BCRHS = np.zeros((Nx+2)*(Ny+2))  # RHS vector
    q = -1  # a counter in case I'm wrong with my count of nonzero elements
    #  assign value to the corner nodes (useless cells)
    q = int_range(0, 3)
    ii[q] = BC.domain.corners
    jj[q] = BC.domain.corners
    s[q] = np.max(BC.top.b/2 + BC.top.a/dy_end)
    BCRHS[BC.domain.corners] = 0.0

    if (not BC.top.periodic) and (not BC.bottom.periodic):
        # top boundary
        j=Ny+1
        i=int_range(1,Nx)
        q = q[-1]+i
        ii[q] = G[i,j]  
        jj[q] = G[i,j]  
        s[q] = (BC.top.b/2 + BC.top.a/(dy_end*rp))
        q = q[-1]+i
        ii[q] = G[i,j]  
        jj[q] = G[i,j-1] 
        s[q] = (BC.top.b/2 - BC.top.a/(dy_end*rp))
        BCRHS[G[i,j]] = (BC.top.c)

        # Bottom boundary
        j=0
        # i=1:Nx already defined
        q = q[-1]+i
        ii[q] = G[i,j]  
        jj[q] = G[i,j+1]  
        s[q] = -(BC.bottom.b/2 + BC.bottom.a/(rp*dy_1))
        q = q[-1]+i
        ii[q] = G[i,j]  
        jj[q] = G[i,j] 
        s[q] = -(BC.bottom.b/2 - BC.bottom.a/(rp*dy_1))
        BCRHS[G[i,j]] = (-BC.bottom.c)
    elif BC.top.periodic or BC.bottom.periodic: # periodic boundary
        # top boundary
        j=Ny+1
        # i=int_range(1,Nx)
        i=int_range(1,Nx)
        q = q[-1]+i
        ii[q] = G[i,j]  
        jj[q] = G[i,j]  
        s[q] = 1
        q = q[-1]+i
        ii[q] = G[i,j]  
        jj[q] = G[i,j-1]  
        s[q] = -1
        q = q[-1]+i
        ii[q] = G[i,j]  
        jj[q] = G[i,0] 
        s[q] = dy_end/dy_1
        q = q[-1]+i
        ii[q] = G[i,j]  
        jj[q] = G[i,1] 
        s[q] = -dy_end/dy_1
        BCRHS[G[i,j]] = 0

        # Bottom boundary
        j=0
        # i=int_range(1,Nx)
        q = q[-1]+i
        ii[q] = G[i,j]  
        jj[q] = G[i,j]  
        s[q] = 1
        q = q[-1]+i
        ii[q] = G[i,j]  
        jj[q] = G[i,j+1]  
        s[q] = 1
        q = q[-1]+i
        ii[q] = G[i,j]
        jj[q] = G[i,Ny+1] 
        s[q] = -1
        q = q[-1]+i
        ii[q] = G[i,j]  
        jj[q] = G[i,Ny+2]
        s[q] = -1
        BCRHS[G[i,j]] = 0

    if (not BC.left.periodic) and (not BC.right.periodic):
        # right boundary
        i = Nx+1  # -1 for python 0-based indexing
        j = int_range(1, Ny)
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[i,j]
        s[q] = BC.right.b/2 + BC.right.a/dx_end
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[i-1,j]
        s[q] = BC.right.b/2 - BC.right.a/dx_end
        BCRHS[G[i,j]] = BC.right.c

        # Left boundary
        i = 0
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[i+1,j]
        s[q] = -(BC.left.b/2 + BC.left.a/dx_1)
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[i,j]
        s[q] = -(BC.left.b/2 - BC.left.a/dx_1)
        BCRHS[G[i,j]] = -BC.left.c
    elif BC.right.periodic or BC.left.periodic:  # periodic boundary condition
        # Right boundary
        i = Nx+1
        j = int_range(1, Ny)
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[i,j]
        s[q] = 1
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[i-1,j]
        s[q] = -1
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[0,j]
        s[q] = dx_end/dx_1
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[1,j]
        s[q] = -dx_end/dx_1
        BCRHS[G[i,j]] = 0
        # Left boundary
        i = 0
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[i,j]
        s[q] = 1.0
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[i+1,j]
        s[q] = 1.0
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[Nx,j]
        s[q] = -1.0
        q = q[-1]+j
        ii[q] = G[i,j]
        jj[q] = G[Nx+1,j]
        s[q] = -1.0
        BCRHS[G[i,j]] = 0.0
    # Build the sparse matrix of the boundary conditions
    q = q[-1] + 1
    BCMatrix = csr_array((s[0:q], (ii[0:q], jj[0:q])), 
                         shape=((Nx+2)*(Ny+2), (Nx+2)*(Ny+2)))
    return BCMatrix, BCRHS


def boundaryConditionCylindrical3D(BC: BoundaryCondition3D):
    # extract data from the mesh structure
    Nx, Ny, Nz = BC.domain.dims
    G=BC.domain.cell_numbers()
    dx_1 = BC.domain.cellsize.x[0]
    dx_end = BC.domain.cellsize.x[-1]
    dy_1 = BC.domain.cellsize.y[0]
    dy_end = BC.domain.cellsize.y[-1]
    dz_1 = BC.domain.cellsize.z[0]
    dz_end = BC.domain.cellsize.z[-1]
    rp = BC.domain.cellcenters.x[:, np.newaxis]

    i_ind = int_range(1,Nx)[:, np.newaxis, np.newaxis]
    j_ind = int_range(1,Ny)[np.newaxis, :, np.newaxis]
    k_ind = int_range(1,Nz)[np.newaxis, np.newaxis, :]
    # number of boundary nodes (axact number is 2[(m+1)(n+1)*(n+1)*(p+1)+(m+1)*p+1]:
    nb = 8*((Nx+1)*(Ny+1)+(Nx+1)*(Nz+1)+(Ny+1)*(Nz+1))

    # define the vectors to be used for the creation of the sparse matrix
    ii = np.zeros(nb, dtype=int)
    jj = np.zeros(nb, dtype=int)
    s = np.zeros(nb, dtype=float)

    # define the RHS column vector
    BCRHS = np.zeros((Nx+2)*(Ny+2)*(Nz+2))

    # assign value to the corner nodes (useless cells)
    q = int_range(0, 7)
    ii[q] = BC.domain.corners 
    jj[q] = BC.domain.corners
    s[q] = 1.0
    BCRHS[BC.domain.corners] = 0.0

    # assign values to the edges (useless cells)
    q = q[-1]+int_range(1, np.size(BC.domain.edges))
    ii[q] = BC.domain.edges 
    jj[q] = BC.domain.edges
    s[q] = 1.0
    BCRHS[BC.domain.edges] = 0.0

    # Assign values to the boundary condition matrix and the RHS vector based
    # on the BC structure
    if (not BC.top.periodic) and (not BC.bottom.periodic):
        # top boundary
        j=Ny+1
        i = i_ind
        k = k_ind
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = (BC.top.b/2 + BC.top.a/(dy_end*rp)).ravel()
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j-1,k].ravel()
        s[q] = (BC.top.b/2 - BC.top.a/(dy_end*rp)).ravel()
        BCRHS[G[i,j,k].ravel()] = (BC.top.c).ravel()

        # Bottom boundary
        j=0
        i=i_ind
        k=k_ind
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j+1,k].ravel()
        s[q] = -(BC.bottom.b/2 + BC.bottom.a/(dy_1*rp)).ravel()
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = -(BC.bottom.b/2 - BC.bottom.a/(dy_1*rp)).ravel()
        BCRHS[G[i,j,k].ravel()] = -(BC.bottom.c).ravel()
    elif BC.top.periodic or BC.bottom.periodic: # periodic
        # top boundary
        j=Ny+1
        i=i_ind
        k=k_ind
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = 1.0
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j-1,k].ravel()
        s[q] = -1.0
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,0,k].ravel()
        s[q] = dy_end/dy_1
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,1,k].ravel()
        s[q] = -dy_end/dy_1
        BCRHS[G[i,j,k].ravel()] = 0.0

        # Bottom boundary
        j=0
        i=i_ind
        k=k_ind
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = 1.0
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j+1,k].ravel()
        s[q] = 1.0
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,Ny,k].ravel()
        s[q] = -1.0
        q = q[-1]+int_range(1,Nx*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,Ny+1,k].ravel()
        s[q] = -1.0
        BCRHS[G[i,j,k].ravel()] = 0.0
    if (not BC.right.periodic) and (not BC.left.periodic):
        # Right boundary
        i=Nx+1
        j=j_ind
        k=k_ind
        q = q[-1]+int_range(1,Ny*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = (BC.right.b/2 + BC.right.a/dx_end).ravel()
        q = q[-1]+int_range(1,Ny*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i-1,j,k].ravel()
        s[q] = (BC.right.b/2 - BC.right.a/dx_end).ravel()
        BCRHS[G[i,j,k].ravel()] = (BC.right.c).ravel()

        # Left boundary
        i = 0
        j=j_ind
        k=k_ind
        q = q[-1]+int_range(1,Ny*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i+1,j,k].ravel()
        s[q] = -(BC.left.b/2 + BC.left.a/dx_1).ravel()
        q = q[-1]+int_range(1,Ny*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = -(BC.left.b/2 - BC.left.a/dx_1).ravel()
        BCRHS[G[i,j,k].ravel()] = -(BC.left.c).ravel()
    elif BC.right.periodic or BC.left.periodic: # periodic
        # Right boundary
        i=Nx+1
        j=j_ind
        k=k_ind
        q = q[-1]+int_range(1,Ny*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = 1.0
        q = q[-1]+int_range(1,Ny*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i-1,j,k].ravel()
        s[q] = -1.0
        q = q[-1]+int_range(1,Ny*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[0,j,k].ravel()
        s[q] = dx_end/dx_1
        q = q[-1]+int_range[1,Ny*Nz]
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[1,j,k].ravel()
        s[q] = -dx_end/dx_1
        BCRHS[G[i,j,k].ravel()] = 0.0

        # Left boundary
        i = 0
        j=j_ind
        k=k_ind
        q = q[-1]+int_range(1,Ny*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = 1.0
        q = q[-1]+int_range(1,Ny*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i+1,j,k].ravel()
        s[q] = 1.0
        q = q[-1]+int_range(1,Ny*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[Nx,j,k].ravel()
        s[q] = -1.0
        q = q[-1]+int_range(1,Ny*Nz)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[Nx+1,j,k].ravel()
        s[q] = -1.0
        BCRHS[G[i,j,k].ravel()] = 0.0
    if (not BC.front.periodic) and (not BC.back.periodic):
        # Front boundary
        k=Nz+1
        i = i_ind
        j = j_ind
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = (BC.front.b/2 + BC.front.a/dz_end).ravel()
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k-1].ravel()
        s[q] = (BC.front.b/2 - BC.front.a/dz_end).ravel()
        BCRHS[G[i,j,k].ravel()] = (BC.front.c).ravel()

        # Back boundary
        k=0
        i = i_ind
        j = j_ind
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k+1].ravel()
        s[q] = -(BC.back.b/2 + BC.back.a/dz_1).ravel()
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = -(BC.back.b/2 - BC.back.a/dz_1).ravel()
        BCRHS[G[i,j,k].ravel()] = -(BC.back.c).ravel()
    elif BC.front.periodic or BC.back.periodic: # periodic
        # Front boundary
        k=Nz+1
        i = i_ind
        j = j_ind
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = 1.0
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k-1].ravel()
        s[q] = -1.0
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,0].ravel()
        s[q] = dz_end/dz_1
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,1].ravel()
        s[q] = -dz_end/dz_1
        BCRHS[G[i,j,k].ravel()] = 0.0

        # Back boundary
        k=0
        i = i_ind
        j = j_ind
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k].ravel()
        s[q] = 1.0
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,k+1].ravel()
        s[q] = 1.0
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,Nz].ravel()
        s[q] = -1.0
        q = q[-1]+int_range(1,Nx*Ny)
        ii[q] = G[i,j,k].ravel()
        jj[q] = G[i,j,Nz+1].ravel()
        s[q] = -1.0
        BCRHS[G[i,j,k].ravel()] = 0.0

    # Build the sparse matrix of the boundary conditions
    q = q[-1]+1
    BCMatrix = csr_array((s[0:q], (ii[0:q], jj[0:q])), 
                         shape=((Nx+2)*(Ny+2)*(Nz+2), (Nx+2)*(Ny+2)*(Nz+2)))
    return BCMatrix, BCRHS


def cellBoundary1D(phi, BC):
    # extract data from the mesh structure
    # Nx = MeshStructure.numberofcells
    dx_1 = BC.domain.cellsize.x[1]
    dx_end = BC.domain.cellsize.x[-1]

    # boundary condition (a d\phi/dx + b \phi = c, a column vector of [d a])
    # a (phi(i)-phi(i-1))/dx + b (phi(i)+phi(i-1))/2 = c
    # phi(i) (a/dx+b/2) + phi(i-1) (-a/dx+b/2) = c
    # Right boundary, i=m+2
    # phi(i) (a/dx+b/2) = c- phi(i-1) (-a/dx+b/2)
    # Left boundary, i=2
    #  phi(i-1) (-a/dx+b/2) = c - phi(i) (a/dx+b/2)
    # define the new phi
    if (not BC.left.periodic) and (not BC.right.periodic):
        phiBC = np.hstack([(BC.left.c-phi[0]*(BC.left.a/dx_1+BC.left.b/2))/(-BC.left.a/dx_1+BC.left.b/2), 
        phi,
        (BC.right.c-phi[-1]*(-BC.right.a/dx_end+BC.right.b/2))/(BC.right.a/dx_end+BC.right.b/2)])
    else:
        phiBC = np.hstack([phi[-1], phi, phi[0]])
    return phiBC


def cellBoundary2D(phi, BC):
    # extract data from the mesh structure
    Nx, Ny = BC.domain.dims
    dx_1 = BC.domain.cellsize.x[0]
    dx_end = BC.domain.cellsize.x[-1]
    dy_1 = BC.domain.cellsize.y[0]
    dy_end = BC.domain.cellsize.y[-1]

    # define the output matrix
    phiBC = np.zeros((Nx+2, Ny+2))
    phiBC[1:Nx+1, 1:Ny+1] = phi

    # Assign values to the boundary values
    if (not BC.top.periodic) and (not BC.bottom.periodic):
        # top boundary
        j=Ny+1
        i = int_range(1, Nx)
        phiBC[i,j]= (BC.top.c-phi[:,-1]*(-BC.top.a/dy_end+BC.top.b/2))/(BC.top.a/dy_end+BC.top.b/2)

        # Bottom boundary
        j=0
        phiBC[i,j]= (BC.bottom.c-phi[:,0]*(BC.bottom.a/dy_1+BC.bottom.b/2))/(-BC.bottom.a/dy_1+BC.bottom.b/2)
    else:
        # top boundary
        j=Ny+1
        i = int_range(1, Nx)
        phiBC[i,j]= phi[:,0]

        # Bottom boundary
        j=0
        phiBC[i,j]= phi[:,-1]

    if (not BC.left.periodic) and (not BC.right.periodic):
        # Right boundary
        i = Nx+1
        j = int_range(1, Ny)
        phiBC[i,j]= (BC.right.c-phi[-1,:]*(-BC.right.a/dx_end+BC.right.b/2))/(BC.right.a/dx_end+BC.right.b/2)

        # Left boundary
        i = 0
        phiBC[i,j]= (BC.left.c-phi[0,:]*(BC.left.a/dx_1+BC.left.b/2))/(-BC.left.a/dx_1+BC.left.b/2)
    else:
        # Right boundary
        i = Nx+1
        j = int_range(1, Ny)
        phiBC[i,j]= phi[0,:]

        # Left boundary
        i = 0
        phiBC[i,j]= phi[-1,:]
    return phiBC


def cellBoundary3D(phi, BC):
    Nx, Ny, Nz = BC.domain.dims
    dx_1 = BC.domain.cellsize.x[0]
    dx_end = BC.domain.cellsize.x[-1]
    dy_1 = BC.domain.cellsize.y[0]
    dy_end = BC.domain.cellsize.y[-1]
    dz_1 = BC.domain.cellsize.z[0]
    dz_end = BC.domain.cellsize.z[-1]

    i_ind = int_range(1,Nx)[:, np.newaxis, np.newaxis]
    j_ind = int_range(1,Ny)[np.newaxis, :, np.newaxis]
    k_ind = int_range(1,Nz)[np.newaxis, np.newaxis, :]
    
    # define the output matrix
    phiBC = np.zeros((Nx+2, Ny+2, Nz+2))
    phiBC[1:Nx+1, 1:Ny+1, 1:Nz+1] = phi

    # Assign values to the boundary values
    if (not BC.top.periodic) and (not BC.bottom.periodic):
        # top boundary
        j=Ny+1
        i = i_ind
        k = k_ind
        phiBC[i,j,k]= ((BC.top.c-phi[:,-1,:]*(-BC.top.a/dy_end+BC.top.b/2))/(BC.top.a/dy_end+BC.top.b/2))[:, np.newaxis, :]

        # Bottom boundary
        j=0
        i = i_ind
        k = k_ind
        phiBC[i,j,k]= ((BC.bottom.c-phi[:,0,:]*(BC.bottom.a/dy_1+BC.bottom.b/2))/(-BC.bottom.a/dy_1+BC.bottom.b/2))[:, np.newaxis, :]
    else:
        # top boundary
        j=Ny+1
        i = i_ind
        k = k_ind
        phiBC[i,j,k]= phi[:,0,:]

        # Bottom boundary
        j=0
        i = i_ind
        k = k_ind
        phiBC[i,j,k]= phi[:,-1,:]

    if (not BC.left.periodic) and (not BC.right.periodic):
        # Right boundary
        i = Nx+1
        j = j_ind
        k = k_ind
        phiBC[i,j,k]= (BC.right.c-phi[-1,:,:]*(-BC.right.a/dx_end+BC.right.b/2))/(BC.right.a/dx_end+BC.right.b/2)

        # Left boundary
        i = 0
        j = j_ind
        k = k_ind
        phiBC[i,j,k]= (BC.left.c-phi[0,:,:]*(BC.left.a/dx_1+BC.left.b/2))/(-BC.left.a/dx_1+BC.left.b/2)
    else:
        # Right boundary
        i = Nx+1
        j = j_ind
        k = k_ind
        phiBC[i,j,k]= phi[0,:,:]

        # Left boundary
        i = 0
        j = j_ind
        k = k_ind
        phiBC[i,j,k]= phi[-1,:,:]

    if (not BC.bottom.periodic) and (not BC.top.periodic):
        # front boundary
        i = i_ind
        j = j_ind
        k = Nz+1
        phiBC[i,j,k]= ((BC.front.c-phi[:,:,-1]*(-BC.front.a/dz_end+BC.front.b/2))/(BC.front.a/dz_end+BC.front.b/2))[:, :, np.newaxis]

        # back boundary
        i = i_ind
        j = j_ind
        k = 0
        phiBC[i,j,k]= ((BC.back.c-phi[:,:,0]*(BC.back.a/dz_1+BC.back.b/2))/(-BC.back.a/dz_1+BC.back.b/2))[:, :, np.newaxis]
    else:
        # front boundary
        i = i_ind
        j = j_ind
        k = Nz+1
        phiBC[i,j,k]= phi[:,:,0]

        # back boundary
        i = i_ind
        j = j_ind
        k = 0
        phiBC[i,j,k]= phi[:,:,-1]
    return phiBC


def cellBoundaryCylindrical3D(phi, BC):
    Nx, Ny, Nz = BC.domain.dims
    dx_1 = BC.domain.cellsize.x[0]
    dx_end = BC.domain.cellsize.x[-1]
    dy_1 = BC.domain.cellsize.y[0]
    dy_end = BC.domain.cellsize.y[-1]
    dz_1 = BC.domain.cellsize.z[0]
    dz_end = BC.domain.cellsize.z[-1]
    rp = BC.domain.cellcenters.x[:, np.newaxis]

    i_ind = int_range(1,Nx)[:, np.newaxis, np.newaxis]
    j_ind = int_range(1,Ny)[np.newaxis, :, np.newaxis]
    k_ind = int_range(1,Nz)[np.newaxis, np.newaxis, :]
    
    # define the output matrix
    phiBC = np.zeros((Nx+2, Ny+2, Nz+2))
    phiBC[1:Nx+1, 1:Ny+1, 1:Nz+1] = phi

    # Assign values to the boundary values
    if (not BC.top.periodic) and (not BC.bottom.periodic):
        # top boundary
        j=Ny+1
        i = i_ind
        k = k_ind
        phiBC[i,j,k]= ((BC.top.c-phi[:,-1,:]*(-BC.top.a/(dy_end*rp)+BC.top.b/2))/(BC.top.a/(dy_end*rp)+BC.top.b/2))[:, np.newaxis, :]

        # Bottom boundary
        j=0
        i = i_ind
        k = k_ind
        phiBC[i,j,k]= ((BC.bottom.c-phi[:,0,:]*(BC.bottom.a/(dy_1*rp)+BC.bottom.b/2))/(-BC.bottom.a/(dy_1*rp)+BC.bottom.b/2))[:, np.newaxis, :]
    else:
        # top boundary
        j=Ny+1
        i = i_ind
        k = k_ind
        phiBC[i,j,k]= phi[:,0,:]

        # Bottom boundary
        j=0
        i = i_ind
        k = k_ind
        phiBC[i,j,k]= phi[:,-1,:]

    if (not BC.left.periodic) and (not BC.right.periodic):
        # Right boundary
        i = Nx+1
        j = j_ind
        k = k_ind
        phiBC[i,j,k]= (BC.right.c-phi[-1,:,:]*(-BC.right.a/dx_end+BC.right.b/2))/(BC.right.a/dx_end+BC.right.b/2)

        # Left boundary
        i = 0
        j = j_ind
        k = k_ind
        phiBC[i,j,k]= (BC.left.c-phi[0,:,:]*(BC.left.a/dx_1+BC.left.b/2))/(-BC.left.a/dx_1+BC.left.b/2)
    else:
        # Right boundary
        i = Nx+1
        j = j_ind
        k = k_ind
        phiBC[i,j,k]= phi[0,:,:]

        # Left boundary
        i = 0
        j = j_ind
        k = k_ind
        phiBC[i,j,k]= phi[-1,:,:]

    if (not BC.bottom.periodic) and (not BC.top.periodic):
        # front boundary
        i = i_ind
        j = j_ind
        k = Nz+1
        phiBC[i,j,k]= ((BC.front.c-phi[:,:,-1]*(-BC.front.a/dz_end+BC.front.b/2))/(BC.front.a/dz_end+BC.front.b/2))[:, :, np.newaxis]

        # back boundary
        i = i_ind
        j = j_ind
        k = 0
        phiBC[i,j,k]= ((BC.back.c-phi[:,:,0]*(BC.back.a/dz_1+BC.back.b/2))/(-BC.back.a/dz_1+BC.back.b/2))[:, :, np.newaxis]
    else:
        # front boundary
        i = i_ind
        j = j_ind
        k = Nz+1
        phiBC[i,j,k]= phi[:,:,0]

        # back boundary
        i = i_ind
        j = j_ind
        k = 0
        phiBC[i,j,k]= phi[:,:,-1]
    return phiBC


def cellBoundaryRadial2D(phi, BC):
    # extract data from the mesh structure
    Nx, Ny = BC.domain.dims
    dx_1 = BC.domain.cellsize.x[0]
    dx_end = BC.domain.cellsize.x[-1]
    dy_1 = BC.domain.cellsize.y[0]
    dy_end = BC.domain.cellsize.y[-1]
    rp = BC.domain.cellcenters.x

    # define the output matrix
    phiBC = np.zeros((Nx+2, Ny+2))
    phiBC[1:Nx+1, 1:Ny+1] = phi

    # Assign values to the boundary values
    if (not BC.top.periodic) and (not BC.bottom.periodic):
        # top boundary
        j=Ny+1
        i = int_range(1, Nx)
        phiBC[i,j]= (BC.top.c-phi[:,-1]*(-BC.top.a/(dy_end*rp)+BC.top.b/2))/(BC.top.a/(dy_end*rp)+BC.top.b/2)

        # Bottom boundary
        j=0
        phiBC[i,j]= (BC.bottom.c-phi[:,0]*(BC.bottom.a/(dy_1*rp)+BC.bottom.b/2))/(-BC.bottom.a/(dy_1*rp)+BC.bottom.b/2)
    else:
        # top boundary
        j=Ny+1
        i = int_range(1, Nx)
        phiBC[i,j]= phi[:,0]

        # Bottom boundary
        j=0
        phiBC[i,j]= phi[:,-1]

    if (not BC.left.periodic) and (not BC.right.periodic):
        # Right boundary
        i = Nx+1
        j = int_range(1, Ny)
        phiBC[i,j]= (BC.right.c-phi[-1,:]*(-BC.right.a/dx_end+BC.right.b/2))/(BC.right.a/dx_end+BC.right.b/2)

        # Left boundary
        i = 0
        phiBC[i,j]= (BC.left.c-phi[0,:]*(BC.left.a/dx_1+BC.left.b/2))/(-BC.left.a/dx_1+BC.left.b/2)
    else:
        # Right boundary
        i = Nx+1
        j = int_range(1, Ny)
        phiBC[i,j]= phi[0,:]

        # Left boundary
        i = 0
        phiBC[i,j]= phi[-1,:]
    return phiBC


def cellBoundary(phi, BC) -> np.ndarray:
    if issubclass(type(BC.domain), Mesh1D):
        return cellBoundary1D(phi, BC)
    elif (type(BC.domain) is Mesh2D) or (type(BC.domain) is MeshCylindrical2D):
        return cellBoundary2D(phi, BC)
    elif (type(BC.domain) is MeshRadial2D):
        return cellBoundaryRadial2D(phi, BC)
    elif (type(BC.domain) is Mesh3D):
        return cellBoundary3D(phi, BC)
    elif (type(BC.domain) is MeshCylindrical3D):
        return cellBoundaryCylindrical3D(phi, BC)
    else:
        raise Exception("The cellBoundary function is not defined for this mesh type.")

        
def boundaryConditionTerm(BC):
    if issubclass(type(BC.domain), Mesh1D):
        return boundaryCondition1D(BC)
    elif (type(BC.domain) is Mesh2D) or (type(BC.domain) is MeshCylindrical2D):
        return boundaryCondition2D(BC)
    elif (type(BC.domain) is MeshRadial2D):
        return boundaryConditionRadial2D(BC)
    elif (type(BC.domain) is Mesh3D):
        return boundaryCondition3D(BC)
    elif (type(BC.domain) is MeshCylindrical3D):
        return boundaryConditionCylindrical3D(BC)
    
    

   