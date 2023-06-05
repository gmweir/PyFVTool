"""
Mesh generation classes and functions


Classes
---------
CellSize - 

CellLocation - 

FaceLocation - 

MeshStructure - 

Mesh1D - 
Mesh2D - 
Mesh3D - 
MeshCylindrical1D - 
MeshSpherical1D - 
MeshCylindrical2D - 
MeshRadial2D - 
MeshCylindrical3D - 
MeshSpherical3D - 

Functions
----------

_facelocation_to_cellsize
_mesh_1d_param
_mesh_2d_param
_mesh_3d_param

createMesh1D
createMesh2D
createMesh3D

createMeshSpherical3D
createMeshRadial2D 
createMeshSpherical1D
createMeshCylindrical3D
createMeshCylindrical2D
createMeshCylindrical1D




"""
import numpy as np
from warnings import warn
from typing import overload
from .utilities import *


class BasicMeshContainer(object):
    """
    Utility class for creating meshes
    """
    
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        self.x = x
        self.y = y
        self.z = z

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
    
class CellSize(BasicMeshContainer):    pass

class CellLocation(BasicMeshContainer):    pass

class FaceLocation(BasicMeshContainer):    pass

    
class MeshStructure(BasicMeshContainer):
    """
    Basic mesh class
       
    Attributes
    ----------
    dimension: {float}
        tag used to define the problem grid. 
            
    dims: {int}
        Number of elements in each coordinate
        
    cellsize: {ndarray(float), ...}
        Physical length/dimension of each cell
    
    cellcenters: {ndarray(float), ...}
        Coordinates of cell nodes in grid
        
    facecenters: {ndarray(float), ...}
        Coordinates of cell faces in grid
        
    corners: {ndarray(int)}
        Index of cell 'corners' 
        
    edges: {ndarray(int)}
        Index of cell 'edges' 
           
    
    Methods
    --------
    visualize: {None}
        NotImplemented - intended for plotting / visualizing the grid
        
    shift_origin: {x: float, y: float, z: float}
        shift entire grid in up to 3-cartesian dimensions
    
    Notes
    ------
    Cell corners / edges are important for boundary conditions in multi-dimensional problems

    dimension = 1:   1D cartesion   (x) 
    dimension = 2:   2D cartesion   (x,y)  
    dimension = 3:   3D cartesion   (x,y,z) 
    dimension = 1.5: 1D cylindrical (r)    -- axisymmetric and uniform along z
    dimension = 2.5: 2D cylindrical (r, z) -- axisymmetric cylinder            
    dimension = 3.2: 3D cylindrical (r, theta, z)
    dimension = 1.8: 1D spherical   (r)
    dimension = 2.8: 2D cylindrical? (r, theta) -- spherical coords?
    dimension = 3.5: 3D spherical   (r, theta, phi)  

    
    See also
    -------
    dimension = 1:   Mesh1D,  createMesh1D
    dimension = 2:   Mesh2D,  createMesh2D  
    dimension = 3:   Mesh3D,  createMesh3D 
    dimension = 1.5: MeshCylindrical1D, createMeshCylindrical1D
    dimension = 2.5: MeshCylindrical2D, createMeshCylindrical2D
    dimension = 3.2: MeshCylindrical3D, createMeshCylindrical3D
    dimension = 1.8: MeshSpherical1D,   createMeshSpherical1D
    dimension = 2.8: MeshRadial2D,      createMeshRadial2D
    dimension = 3.5: MeshSpherical3D ,  createMeshSpherical3D
    
    
    """
    
    def __init__(self, dimension, dims, cellsize,
                 cellcenters, facecenters, corners, edges) -> None:
        self.dimension = dimension
        self.dims = dims
        self.cellsize = cellsize
        self.cellcenters = cellcenters
        self.facecenters = facecenters
        self.corners = corners
        self.edges = edges
        
    def visualize(self):
        pass
    
    def shift_origin(self, x=0.0, y=0.0, z=0.0):
        self.cellcenters.x += x
        self.cellcenters.y += y
        self.cellcenters.z += z
        self.facecenters.x += x
        self.facecenters.y += y
        self.facecenters.z += z


class Mesh1D(MeshStructure):
    def __init__(self, dims, cell_size, cell_location, face_location, corners, edges):
        dimension = 1

        super(Mesh1D, self).__init__(        
            dimension=dimension, dims=dims, cellsize=cell_size, cellcenters=cell_location, facecenters=face_location, corners=corners, edges=edges)

    def cell_numbers(self):
        Nx = self.dims[0]
        return int_range(0, Nx+1)

    def __repr__(self):
        print(f"1D Cartesian mesh with {self.dims[0]} cells")
        return ""


class Mesh2D(MeshStructure):
    def __init__(self, dims, cell_size, cell_location, face_location, corners, edges):
        dimension = 2

        super(Mesh2D, self).__init__(        
            dimension=dimension, dims=dims, cellsize=cell_size, cellcenters=cell_location, facecenters=face_location, corners=corners, edges=edges)


    def cell_numbers(self):
        Nx, Ny = self.dims
        G = int_range(0, (Nx+2)*(Ny+2)-1)
        return G.reshape(Nx+2, Ny+2)

    def __repr__(self):
        print(f"2D Cartesian mesh with {self.dims[0]}x{self.dims[1]} cells")
        return ""


class Mesh3D(MeshStructure):
    def __init__(self, dims, cell_size, cell_location, face_location, corners, edges):
        dimension = 3
        
        super(Mesh3D, self).__init__(        
            dimension=dimension, dims=dims, cellsize=cell_size, cellcenters=cell_location, facecenters=face_location, corners=corners, edges=edges)


    def cell_numbers(self):
        Nx, Ny, Nz = self.dims
        G = int_range(0, (Nx+2)*(Ny+2)*(Nz+2)-1)
        return G.reshape(Nx+2, Ny+2, Nz+2)

    def __repr__(self):
        print(
            f"3D Cartesian mesh with Nx={self.dims[0]}xNy={self.dims[1]}xNz={self.dims[1]} cells")
        return ""


class MeshCylindrical1D(Mesh1D):
    def __init__(self, dims, cell_size, cell_location, face_location, corners, edges):
        dimension = 1.5

        # Note:  Keeping with old-style initialization (no super call) to avoid overwriting the dimensional represetation of the this mesh. 
        MeshStructure.__init__(
            self, dimension=dimension, dims=dims, cellsize=cell_size, cellcenters=cell_location, facecenters=face_location, corners=corners, edges=edges)
        
    def __repr__(self):
        print(f"1D Cylindrical (radial) mesh with Nr={self.dims[0]} cells")
        return ""


class MeshSpherical1D(Mesh1D):
    def __init__(self, dims, cell_size, cell_location, face_location, corners, edges):
        dimension = 1.8

        # Note:  Keeping with old-style initialization (no super call) to avoid overwriting the dimensional represetation of the this mesh. 
        MeshStructure.__init__(
            self, dimension=dimension, dims=dims, cellsize=cell_size, cellcenters=cell_location, facecenters=face_location, corners=corners, edges=edges)

    def __repr__(self):
        print(f"1D Spherical mesh with Nr={self.dims[0]} cells")
        return ""


class MeshCylindrical2D(Mesh2D):
    def __init__(self, dims, cell_size, cell_location, face_location, corners, edges):
        dimension = 2.5
        
        # Note:  Keeping with old-style initialization (no super call) to avoid overwriting the dimensional represetation of the this mesh. 
        MeshStructure.__init__(
            self, dimension=dimension, dims=dims, cellsize=cell_size, cellcenters=cell_location, facecenters=face_location, corners=corners, edges=edges)

    def __repr__(self):
        print(
            f"2D Cylindrical mesh with Nr={self.dims[0]}xNz={self.dims[1]} cells")
        return ""


class MeshRadial2D(Mesh2D):
    def __init__(self, dims, cell_size, cell_location, face_location, corners, edges):
        dimension = 2.8
        
        # Note:  Keeping with old-style initialization (no super call) to avoid overwriting the dimensional represetation of the this mesh. 
        MeshStructure.__init__(
            self, dimension=dimension, dims=dims, cellsize=cell_size, cellcenters=cell_location, facecenters=face_location, corners=corners, edges=edges)

    def __repr__(self):
        print(
            f"2D Radial mesh with Nr={self.dims[0]}xN_theta={self.dims[1]} cells")
        return ""


class MeshCylindrical3D(Mesh3D):
    def __init__(self, dims, cell_size, cell_location, face_location, corners, edges):
        dimension = 3.2

        # Note:  Keeping with old-style initialization (no super call) to avoid overwriting the dimensional represetation of the this mesh. 
        MeshStructure.__init__(
            self, dimension=dimension, dims=dims, cellsize=cell_size, cellcenters=cell_location, facecenters=face_location, corners=corners, edges=edges)

    def __repr__(self):
        print(
            f"3D Cylindrical mesh with Nr={self.dims[0]}xN_theta={self.dims[1]}xNz={self.dims[1]} cells")
        return ""


class MeshSpherical3D(Mesh3D):
    def __init__(self, dims, cell_size, cell_location, face_location, corners, edges):
        dimension = 3.5
        
        # Note:  Keeping with old-style initialization (no super call) to avoid overwriting the dimensional represetation of the this mesh. 
        MeshStructure.__init__(
            self, dimension=dimension, dims=dims, cellsize=cell_size, cellcenters=cell_location, facecenters=face_location, corners=corners, edges=edges)

    def __repr__(self):
        print(
            f"3D Shperical mesh with Nr={self.dims[0]}xN_theta={self.dims[1]}xN_phi={self.dims[1]} cells")
        return ""


def _facelocation_to_cellsize(facelocation):
    """ internal utility function that converts facelocation data (coordinates in real-space) to cell size information """
    return np.hstack([facelocation[1]-facelocation[0],
                      facelocation[1:]-facelocation[0:-1],
                      facelocation[-1]-facelocation[-2]])


def _mesh_1d_param(*args):
    """ an internal utility function that parses input to 1D meshes and produces desired grid parameters """
    if len(args) == 1:
        # Use face locations
        facelocationX = args[0]
        Nx = facelocationX.size-1
        cell_size_x = np.hstack([facelocationX[1]-facelocationX[0],
                                 facelocationX[1:]-facelocationX[0:-1],
                                 facelocationX[-1]-facelocationX[-2]])
        cell_size = CellSize(cell_size_x, np.array([0.0]), np.array([0.0]))
        cell_location = CellLocation(
            0.5*(facelocationX[1:]+facelocationX[0:-1]), np.array([0.0]), np.array([0.0]))
        face_location = FaceLocation(
            facelocationX, np.array([0.0]), np.array([0.0]))
    elif len(args) == 2:
        # Use number of cells and domain length
        Nx = args[0]
        Width = args[1]
        dx = Width/Nx
        cell_size = CellSize(
            dx*np.ones(Nx+2), np.array([0.0]), np.array([0.0]))
        cell_location = CellLocation(
            int_range(1, Nx)*dx-dx/2,
            np.array([0.0]),
            np.array([0.0]))
        face_location = FaceLocation(
            int_range(0, Nx)*dx,
            np.array([0.0]),
            np.array([0.0]))
    dims = np.array([Nx], dtype=int)
    cellsize = cell_size
    cellcenters = cell_location
    facecenters = face_location
    corners = np.array([1], dtype=int)
    edges = np.array([1], dtype=int)
    return dims, cellsize, cellcenters, facecenters, corners, edges


def _mesh_2d_param(*args):
    """ an internal utility function that parses input to 2D meshes and produces desired grid parameters """
    if len(args) == 2:
        # Use face locations
        facelocationX = args[0]
        facelocationY = args[1]
        Nx = facelocationX.size-1
        Ny = facelocationY.size-1
        cell_size = CellSize(_facelocation_to_cellsize(facelocationX),
                             _facelocation_to_cellsize(facelocationY),
                             np.array([0.0]))
        cell_location = CellLocation(
            0.5*(facelocationX[1:]+facelocationX[0:-1]),
            0.5*(facelocationY[1:]+facelocationY[0:-1]),
            np.array([0.0]))
        face_location = FaceLocation(
            facelocationX,
            facelocationY,
            np.array([0.0]))
    elif len(args) == 4:
        # Use number of cells and domain length
        Nx = args[0]
        Ny = args[1]
        Width = args[2]
        Height = args[3]
        dx = Width/Nx
        dy = Height/Ny
        cell_size = CellSize(
            dx*np.ones(Nx+2),
            dy*np.ones(Ny+2),
            np.array([0.0]))
        cell_location = CellLocation(
            int_range(1, Nx)*dx-dx/2,
            int_range(1, Ny)*dy-dy/2,
            np.array([0.0]))
        face_location = FaceLocation(
            int_range(0, Nx)*dx,
            int_range(0, Ny)*dy,
            np.array([0.0]))

    dims = np.array([Nx, Ny], dtype=int)
    cellsize = cell_size
    cellcenters = cell_location
    facecenters = face_location
    G = int_range(1, (Nx+2)*(Ny+2))-1
    corners = G.reshape(Nx+2, Ny+2)[[0, -1, 0, -1], [0, 0, -1, -1]]
    edges = np.array([1], dtype=int)
    return dims, cellsize, cellcenters, facecenters, corners, edges


def _mesh_3d_param(*args):
    """ an internal utility function that parses input to 3D meshes and produces desired grid parameters """
    if len(args) == 3:
        # Use face locations
        facelocationX = args[0]
        facelocationY = args[1]
        facelocationZ = args[2]
        Nx = facelocationX.size-1
        Ny = facelocationY.size-1
        Nz = facelocationZ.size-1
        cell_size = CellSize(_facelocation_to_cellsize(facelocationX),
                             _facelocation_to_cellsize(facelocationY),
                             _facelocation_to_cellsize(facelocationZ))
        cell_location = CellLocation(
            0.5*(facelocationX[1:]+facelocationX[0:-1]),
            0.5*(facelocationY[1:]+facelocationY[0:-1]),
            0.5*(facelocationZ[1:]+facelocationZ[0:-1]))
        face_location = FaceLocation(
            facelocationX,
            facelocationY,
            facelocationZ)
    elif len(args) == 6:
        # Use number of cells and domain length
        Nx = args[0]
        Ny = args[1]
        Nz = args[2]
        Width = args[3]
        Height = args[4]
        Depth = args[5]
        dx = Width/Nx
        dy = Height/Ny
        dz = Depth/Nz
        cell_size = CellSize(
            dx*np.ones(Nx+2),
            dy*np.ones(Ny+2),
            dz*np.ones(Nz+2))
        cell_location = CellLocation(
            int_range(1, Nx)*dx-dx/2,
            int_range(1, Ny)*dy-dy/2,
            int_range(1, Nz)*dz-dz/2)
        face_location = FaceLocation(
            int_range(0, Nx)*dx,
            int_range(0, Ny)*dy,
            int_range(0, Nz)*dz)
    G = int_range(1, (Nx+2)*(Ny+2)*(Nz+2))-1
    G = G.reshape(Nx+2, Ny+2, Nz+2)
    dims = np.array([Nx, Ny, Nz], dtype=int)
    cellsize = cell_size
    cellcenters = cell_location
    facecenters = face_location
    corners = G[np.ix_((0, -1), (0, -1), (0, -1))].flatten()
    edges = np.hstack([G[0, [0, -1], 1:-1].flatten(),
                       G[-1, [0, -1], 1:-1].flatten(),
                       G[0, 1:-1, [0, -1]].flatten(),
                       G[-1, 1:-1, [0, -1]].flatten(),
                       G[1:-1, 0, [0, -1]].flatten(),
                       G[1:-1, -1, [0, -1]].flatten()])
    return dims, cellsize, cellcenters, facecenters, corners, edges


@overload
def createMesh1D(Nx: int, Lx: float) -> Mesh1D:
    ...


@overload
def createMesh1D(face_locations: np.ndarray) -> Mesh1D:
    ...


def createMesh1D(*args) -> Mesh1D:
    """
    An overloaded function that creates a Mesh-structure given basic grid information.
    
    Parameters
    -------           
    Nx: {int}
        Number of grid-points in the x-direction

    Lx: {float}
        Physical length of the grid (has units)

    face_locations: {ndarray}, optional alternative to (Nx, Lx)
        A mesh can be created from the location of the cell faces.
        
    Returns
    -------                   
    out - {MeshStructure object}
        returns a Mesh1D structure for the desired grid
    
    Notes
    -------
    
    Examples
    -------
    >>> m = createMesh1D(Nx=int(10), Lx=float(1.0));   print(m)
    dimension : 1
    dims : [10]
    cellsize : x : [0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]
    y : [0.]
    z : [0.]

    cellcenters : x : [0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95]
    y : [0.]
    z : [0.]

    facecenters : x : [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ]
    y : [0.]
    z : [0.]

    corners : [1]
    edges : [1]
    
    >>> m = createMesh1D(face_locations); print(m)
    ...
    """
    dims, cellsize, cellcenters, facecenters, corners, edges = _mesh_1d_param(
        *args)
    return Mesh1D(dims, cellsize, cellcenters, facecenters, corners, edges)


@overload
def createMesh2D(Nx: int, Ny: int, Lx: float, Ly: float) -> Mesh2D:
    ...


@overload
def createMesh2D(face_locationsX: np.ndarray,
                 face_locationsY: np.ndarray) -> Mesh2D:
    ...


def createMesh2D(*args) -> Mesh2D:
    """
    An overloaded function that creates a Mesh-structure given basic grid information.
    
    Parameters
    -------           
    Nx: {int}
        Number of grid-points in the x-direction

    Ny: {int}
        Number of grid-points in the y-direction

    Lx: {float}
        Physical length of the grid in x-direction (has units)

    Ly: {float}
        Physical length of the grid in y-direction (has units)

    face_locationsX: {ndarray}, optional alternative to (Nx, Ny, Lx, Ly)
        A mesh can be created from the X-location of the cell faces. Paired with face_locationsY input.

    face_locationsY: {ndarray}, optional alternative to (Nx, Ny, Lx, Ly)
        A mesh can be created from the Y-location of the cell faces. Paired with face_locationsX input.


    Returns
    -------                   
    out - {MeshStructure object}
        returns a Mesh2D structure for the desired grid

    Examples
    -------
    >>> m = createMesh2D(Nx=int(10), Ny=int(5), Lx=float(1.0), Ly=float(15.0))
    ...
    
    >>> m = createMesh2D(face_locationsX, face_locationsY)
    ...

    """

    dims, cellsize, cellcenters, facecenters, corners, edges = _mesh_2d_param(
        *args)
    return Mesh2D(dims, cellsize, cellcenters, facecenters, corners, edges)


@overload
def createMesh3D(Nx: int, Ny: int, Nz: int,
                 Lx: float, Ly: float, Lz: float) -> Mesh3D:
    ...


@overload
def createMesh3D(face_locationsX: np.ndarray,
                 face_locationsY: np.ndarray, face_locationsZ: np.ndarray) -> Mesh3D:
    ...


def createMesh3D(*args) -> Mesh3D:
    """
    An overloaded function that creates a Mesh-structure given basic grid information.
    
    Parameters
    -------           
    Nx: {int}
        Number of grid-points in the x-direction

    Ny: {int}
        Number of grid-points in the y-direction

    Nz: {int}
        Number of grid-points in the z-direction

    Lx: {float}
        Physical length of the grid in x-direction (has units)

    Ly: {float}
        Physical length of the grid in y-direction (has units)

    Lz: {float}
        Physical length of the grid in z-direction (has units)

    face_locationsX: {ndarray}, optional alternative to (Nx, Ny, Nz, Lx, Ly, Lz)
        A mesh can be created from the X-location of the cell faces. Paired with face_locationsY/Z input.

    face_locationsY: {ndarray}, optional alternative to (Nx, Ny, Nz, Lx, Ly, Lz)
        A mesh can be created from the Y-location of the cell faces. Paired with face_locationsX/Z input.

    face_locationsZ: {ndarray}, optional alternative to (Nx, Ny, Nz, Lx, Ly, Lz)
        A mesh can be created from the Z-location of the cell faces. Paired with face_locationsX/Y input.


    Returns
    -------                   
    out - {MeshStructure object}
        returns a Mesh3D structure for the desired grid

    Examples
    -------
    >>> m = createMesh3D(Nx=int(10), Ny=int(5), Nz=int(50), Lx=float(1.0), Ly=float(15.0), Lz=float(1.0))
    ...
    
    >>> m = createMesh2D(face_locationsX, face_locationsY, face_locationsZ)
    ...

    """

    dims, cellsize, cellcenters, facecenters, corners, edges = _mesh_3d_param(
        *args)
    return Mesh3D(dims, cellsize, cellcenters, facecenters, corners, edges)


@overload
def createMeshCylindrical1D(Nx: int, Lx: float) -> MeshCylindrical1D:
    ...


@overload
def createMeshCylindrical1D(face_locations: np.ndarray) -> MeshCylindrical1D:
    ...


def createMeshCylindrical1D(*args) -> MeshCylindrical1D:
    """
    An overloaded function that creates a Mesh-structure given basic grid information.
    
    Parameters
    -------           
    Nx: {int}
        Number of grid-points in the r-direction

    Lx: {float}
        Physical length of the grid in r-direction (has units)

    face_locations: {ndarray}, optional alternative to (Nx, Ny, Nz, Lx, Ly, Lz)
        A mesh can be created from the X-location of the cell faces. Paired with face_locationsY/Z input.


    Returns
    -------                   
    out - {MeshStructure object}
        returns a MeshCylindrical1D structure for the desired grid

    Examples
    -------
    >>> m = createMeshCylindrical1D(Nx=int(10), Lx=float(1.0))
    ...
    
    >>> m = createMeshCylindrical1D(face_locations)
    ...
    """
    dims, cellsize, cellcenters, facecenters, corners, edges = _mesh_1d_param(
        *args)
    return MeshCylindrical1D(dims, cellsize, cellcenters, facecenters, corners, edges)


@overload
def createMeshCylindrical2D(Nx: int, Ny: int,
                            Lx: float, Ly: float) -> MeshCylindrical2D:
    ...


@overload
def createMeshCylindrical2D(face_locationsX: np.ndarray,
                            face_locationsY: np.ndarray) -> MeshCylindrical2D:
    ...


def createMeshCylindrical2D(*args) -> MeshCylindrical2D:
    """
    An overloaded function that creates a Mesh-structure given basic grid information.
    
    Parameters
    -------           
    Nx: {int}
        Number of grid-points in the radial-direction

    Ny: {int}
        Number of grid-points in the z-direction

    Lx: {float}
        Physical length of the grid in radial-direction (has units)

    Ly: {float}
        Physical length of the grid in z-direction (has units)

    face_locationsX: {ndarray}, optional alternative to (Nx, Ny, Lx, Ly)
        A mesh can be created from the radial-location of the cell faces. Paired with face_locationsY input.

    face_locationsY: {ndarray}, optional alternative to (Nx, Ny, Lx, Ly)
        A mesh can be created from the Z-location of the cell faces. Paired with face_locationsX input.


    Returns
    -------                   
    out - {MeshStructure object}
        returns a MeshCylindrical2D structure for the desired grid

    Examples
    -------
    >>> m = createMeshCylindrical2D(Nx=int(10), Ny=int(5), Lx=float(1.0), Ly=float(15.0))
    ...
    
    >>> m = createMeshCylindrical2D(face_locationsX, face_locationsY)
    ...                
    """
    dims, cellsize, cellcenters, facecenters, corners, edges = _mesh_2d_param(
        *args)
    return MeshCylindrical2D(dims, cellsize, cellcenters, facecenters, corners, edges)


@overload
def createMeshCylindrical3D(Nx: int, Ny: int,
                            Nz: int, Lx: float,
                            Ly: float, Lz: float) -> MeshCylindrical3D:
    ...


@overload
def createMeshCylindrical3D(face_locationsX: np.ndarray,
                            face_locationsY: np.ndarray,
                            face_locationsZ: np.ndarray) -> MeshCylindrical3D:
    ...


def createMeshCylindrical3D(*args) -> MeshCylindrical3D:
    """
    An overloaded function that creates a Mesh-structure given basic grid information.
    
    Parameters
    -------           
    Nx: {int}
        Number of grid-points in the radial-direction

    Ny: {int}
        Number of grid-points in the polar-direction

    Nz: {int}
        Number of grid-points in the z-direction

    Lx: {float}
        Physical length of the grid in radial-direction (has units)

    Ly: {float}
        Physical length of the grid in polar-direction (has units)

    Lz: {float}
        Physical length of the grid in z-direction (has units)

    face_locationsX: {ndarray}, optional alternative to (Nx, Ny, Nz, Lx, Ly, Lz)
        A mesh can be created from the radial-location of the cell faces. Paired with face_locationsY/Z input.

    face_locationsY: {ndarray}, optional alternative to (Nx, Ny, Nz, Lx, Ly, Lz)
        A mesh can be created from the polar-location of the cell faces. Paired with face_locationsX/Z input.

    face_locationsZ: {ndarray}, optional alternative to (Nx, Ny, Nz, Lx, Ly, Lz)
        A mesh can be created from the Z-location of the cell faces. Paired with face_locationsX/Y input.


    Returns
    -------                   
    out - {MeshStructure object}
        returns a MeshCylindrical3D structure for the desired grid

    Examples
    -------
    >>> m = createMeshCylindrical3D(Nx=int(10), Ny=int(5), Nz=int(50), Lx=float(1.0), Ly=float(15.0), Lz=float(1.0))
    ...
    
    >>> m = createMeshCylindrical2D(face_locationsX, face_locationsY, face_locationsZ)
    ...
    """
    if len(args) == 3:
        theta_max = args[1][-1]
    else:
        theta_max = args[4]
    if theta_max > 2*np.pi:
        warn("Recreate the mesh with an upper bound of 2*pi for theta or there will be unknown consequences!")
    dims, cellsize, cellcenters, facecenters, corners, edges = _mesh_3d_param(
        *args)
    return MeshCylindrical3D(dims, cellsize, cellcenters, facecenters, corners, edges)


@overload
def createMeshSpherical1D(Nx: int, Lx: float) -> MeshSpherical1D:
    ...

@overload
def createMeshSpherical1D(face_locations: np.ndarray) -> MeshSpherical1D:
    ...


def createMeshSpherical1D(*args) -> MeshSpherical1D:
    """
    An overloaded function that creates a Mesh-structure given basic grid information.
    
    Parameters
    -------           
    Nx: {int}
        Number of grid-points in the radial-direction

    Lx: {float}
        Physical length of the grid in radial-direction (has units)

    face_locationsX: {ndarray}, optional alternative to (Nx, Ny, Nz, Lx, Ly, Lz)
        A mesh can be created from the radial-location of the cell faces. 
        

    Returns
    -------                   
    out - {MeshStructure object}
        returns a Mesh3D structure for the desired grid

    Examples
    -------
    >>> m = createMeshSpherical1D(Nx=int(10), Lx=float(1.0))
    ...
    
    >>> m = createMesh2D(face_locationsX)
    ...
    """
    dims, cellsize, cellcenters, facecenters, corners, edges = _mesh_1d_param(
        *args)
    return MeshSpherical1D(dims, cellsize, cellcenters, facecenters, corners, edges)


@overload
def createMeshRadial2D(Nx: int, Ny: int, Lx: float, Ly: float) -> MeshRadial2D:
    ...


@overload
def createMeshRadial2D(face_locationsX: np.ndarray,
                       face_locationsY: np.ndarray) -> MeshRadial2D:
    ...


def createMeshRadial2D(*args) -> MeshRadial2D:
    """
    An overloaded function that creates a Mesh-structure given basic grid information.
    
    Parameters
    -------           
    Nx: {int}
        Number of grid-points in the radial-direction

    Ny: {int}
        Number of grid-points in the polar angle-direction

    Lx: {float}
        Physical length of the grid in radial-direction (has units)

    Ly: {float}
        Physical length of the grid in polar angle-direction (has units)

    face_locationsX: {ndarray}, optional alternative to (Nx, Ny, Lx, Ly)
        A mesh can be created from the radial-location of the cell faces. Paired with face_locationsY input.

    face_locationsY: {ndarray}, optional alternative to (Nx, Ny, Lx, Ly)
        A mesh can be created from the polar angle-location of the cell faces. Paired with face_locationsX input.
        

    Returns
    -------                   
    out - {MeshStructure object}
        returns a MeshRadial2D structure for the desired grid

    Examples
    -------
    >>> m = createMeshRadial2D(Nx=int(10), Ny=int(5), Lx=float(1.0), Ly=float(2.0*numpy.pi))
    ...
    
    >>> m = createMeshRadial2D(face_locationsX, face_locationsY)
    ...    
    """
    if len(args) == 2:
        theta_max = args[1][-1]
    else:
        theta_max = args[3]
    if theta_max > 2*np.pi:
        warn("Recreate the mesh with an upper bound of 2*pi for \theta or there will be unknown consequences!")
    dims, cellsize, cellcenters, facecenters, corners, edges = _mesh_2d_param(
        *args)
    return MeshRadial2D(dims, cellsize, cellcenters, facecenters, corners, edges)


@overload
def createMeshSpherical3D(Nx: int, Ny: int, Nz: int,
                          Lx: float, Ly: float, Lz: float) -> MeshSpherical3D:
    ...

@overload
def createMeshSpherical3D(face_locationsX: np.ndarray, face_locationsY: np.ndarray,
                          face_locationsZ: np.ndarray) -> MeshSpherical3D:
    ...


def createMeshSpherical3D(*args) -> MeshSpherical3D:
    """
    An overloaded function that creates a Mesh-structure given basic grid information.
    
    Parameters
    -------           
    Nx: {int}
        Number of grid-points in the radial-direction

    Ny: {int}
        Number of grid-points in the polar angle-direction

    Nz: {int}
        Number of grid-points in the azimuthal angle-direction

    Lx: {float}
        Physical length of the grid in radial-direction (has units)

    Ly: {float}
        Physical length of the grid in polar angle-direction (has units)

    Lz: {float}
        Physical length of the grid in azimuthal angle-direction (has units)

    face_locationsX: {ndarray}, optional alternative to (Nx, Ny, Nz, Lx, Ly, Lz)
        A mesh can be created from the radial-location of the cell faces. Paired with face_locationsY/Z input.

    face_locationsY: {ndarray}, optional alternative to (Nx, Ny, Nz, Lx, Ly, Lz)
        A mesh can be created from the polar angle-location of the cell faces. Paired with face_locationsX/Z input.

    face_locationsZ: {ndarray}, optional alternative to (Nx, Ny, Nz, Lx, Ly, Lz)
        A mesh can be created from the azimuthal angle-location of the cell faces. Paired with face_locationsX/Y input.


    Returns
    -------                   
    out - {MeshStructure object}
        returns a MeshSpherical3D structure for the desired grid

    Examples
    -------
    >>> m = createMeshSpherical3D(Nx=int(10), Ny=int(5), Nz=int(50), Lx=float(1.0), Ly=float(2.0*numpy.pi), Lz=float(1.0))
    ...
    
    >>> m = createMeshSpherical3D(face_locationsX, face_locationsY, face_locationsZ)
    ...
    """
    if args[4] > 2*np.pi:
        warn("Recreate the mesh with an upper bound of 2*pi for \theta or there will be unknown consequences!")
    if args[5] > 2*np.pi:
        warn("Recreate the mesh with an upper bound of 2*pi for \phi or there will be unknown consequences!")
    dims, cellsize, cellcenters, facecenters, corners, edges = _mesh_3d_param(
        *args)
    return MeshSpherical3D(dims, cellsize, cellcenters, facecenters, corners, edges)
