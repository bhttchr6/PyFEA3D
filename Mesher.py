import gmsh
import numpy as np
import meshio

def get_order(ele_type):
    if ele_type == 'HEX8':
        order = 1
    elif ele_type == 'HEX27':
        order = 2

    return order


def get_meshio_cell_type(ele_type):
    """Reference:
    https://github.com/nschloe/meshio/blob/9dc6b0b05c9606cad73ef11b8b7785dd9b9ea325/src/meshio/xdmf/common.py#L36
    """
    if ele_type == 'TET4':
        cell_type = 'tetra'
    elif ele_type == 'TET10':
        cell_type = 'tetra10'
    elif ele_type == 'HEX8':
        cell_type = 'hexahedron'
    elif ele_type == 'HEX27':
        cell_type = 'hexahedron27'
    elif ele_type == 'HEX20':
        cell_type = 'hexahedron20'
    elif ele_type == 'TRI3':
        cell_type = 'triangle'
    elif ele_type == 'TRI6':
        cell_type = 'triangle6'
    elif ele_type == 'QUAD4':
        cell_type = 'quad'
    elif ele_type == 'QUAD8':
        cell_type = 'quad8'
    else:
        raise NotImplementedError
    return cell_type


def GenerateMesh(Lx, Ly, Lz, Nx, Ny, Nz):
    offset_x = 0.
    offset_y = 0.
    offset_z = 0.
    ele_type = 'HEX8'
    cell_type = get_meshio_cell_type(ele_type)
    degree = get_order(ele_type)

    domain_x = Lx
    domain_y = Ly
    domain_z = Lz

    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)  # save in old MSH format
    if cell_type.startswith('tetra'):
        Rec2d = False  # tris or quads
        Rec3d = False  # tets, prisms or hexas
    else:
        Rec2d = True
        Rec3d = True
    p = gmsh.model.geo.addPoint(offset_x, offset_y, offset_z)

    l = gmsh.model.geo.extrude([(0, p)], domain_x, 0, 0, [Nx], [1])
    s = gmsh.model.geo.extrude([l[1]], 0, domain_y, 0, [Ny], [1], recombine=Rec2d)
    v = gmsh.model.geo.extrude([s[1]], 0, 0, domain_z, [Nz], [1], recombine=Rec3d)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(degree)
    gmsh.write("data/box.msh")
    gmsh.finalize()
    return cell_type, ele_type

def meshIO_read(cell_type, ele_type):
    mesh = meshio.read("data/box.msh")
    points = mesh.points # (num_total_nodes, dim)
    cells =  mesh.cells_dict[cell_type] # (num_cells, num_nodes)
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells})
    nnodes, ndim = points.shape
    nel, nnodes_per_elem = cells.shape
    return nnodes, ndim, nel, nnodes_per_elem, points, cells, mesh

def meshIO_write(Uhat, mesh):
    ele_type = 'HEX8'
    cell_type = get_meshio_cell_type(ele_type)
    points = mesh.points  # (num_total_nodes, dim)
    cells = mesh.cells_dict[cell_type]  # (num_cells, num_nodes)
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells})
    nnodes, ndim = points.shape
    nel, nnodes_per_elem = cells.shape
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    U_mat = Uhat.cpu().detach().numpy().reshape((nnodes, ndim))
    X_new = X + U_mat[:, 0]
    Y_new = Y + U_mat[:, 1]
    Z_new = Z + U_mat[:, 2]

    ## Create deformed mesh and write as gmsh file
    points_new = np.reshape(np.hstack([X_new, Y_new, Z_new]), (nnodes, ndim), order='F')
    out_mesh = meshio.Mesh(points=points_new, cells={cell_type: cells})
    out_mesh.write("data/disp_mesh.vtk")

    mesh = meshio.read("data/disp_mesh.vtk")
    meshio.write("data/disp_mesh.msh", mesh, file_format="gmsh22")