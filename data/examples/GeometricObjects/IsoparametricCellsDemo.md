### Description

This example shows the isoparametric cells supported by the VTK. These cells are nonlinear and contain one or more
mid-side vertices. Isoparametric elements are typically used
in [finite element analysis](https://en.wikipedia.org/wiki/Finite_element_method). The term isoparametric is derived
from the use of the same shape functions (or interpolation functions)  to define the element's geometric shape as are
used to define the displacements within the element.

Options are provided to show a wire frame (`-w`) or to add a back face color (`-b`). You can also remove the plinth with the (`-n`) option.
If you want a single object, use `-o` followed by the object number.

This example illustrates each cell's representation using its parametric coordinates (pcoords) as the vertices of the
cell. In practice, the vertices will correspond to physical points in a finite element model. Use vtkTessellatorFilter
to better see the shape of the cell. See for example, [QuadraticHexahedronDemo](../QuadraticHexahedronDemo)
and [QuadraticTetraDemo](../QuadraticTetraDemo).
