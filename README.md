# Symfem: a symbolic finite element definition library

## Installing Symfem
Symfem can be installed by downloading the repo and running:

```bash
python3 setup.py install
```

Alternatively, the latest release can be installed by running:

```bash
pip3 install symfem
```

## Using Symfem

### Finite elements
Finite elements can be created in symfem using the `create_element` function. For example:

```python
import symfem

lagrange = symfem.create_element("triangle", "Lagrange", 1)
rt = symfem.create_element("tetrahedron", "Raviart-Thomas", 2)
nedelec = symfem.create_element("triangle", "N2curl", 1)
qcurl = symfem.create_element("quadrilateral", "Qcurl", 2)
```

Once an element is created, its polynomial basis can be obtained:

```python
import symfem

lagrange = symfem.create_element("triangle", "Lagrange", 1)
print(lagrange.get_polynomial_basis())
```
```
[1, x, y]
```

The functionals that define the DOFs of the space can be obtained:

```python
import symfem

lagrange = symfem.create_element("triangle", "Lagrange", 1)
print(lagrange.dofs)
```
```
[<symfem.core.functionals.PointEvaluation object at 0x7fd6e5ad1970>, <symfem.core.functionals.PointEvaluation object at 0x7fd6e5ad1bb0>, <symfem.core.functionals.PointEvaluation object at 0x7fd6e5ad1850>]
```

The basis functions spanning the finite element space can be obtained, or tabulated
at a set of points:

```python
import symfem

lagrange = symfem.create_element("triangle", "Lagrange", 1)
print(lagrange.get_basis_functions())

points = [[0, 0], [0.5, 0], [1, 0], [0.25, 0.25]]
print(lagrange.tabulate_basis(points))
```
```
[-x - y + 1, x, y]
[[1, 0, 0], [0.500000000000000, 0.500000000000000, 0], [0, 1, 0], [0.500000000000000, 0.250000000000000, 0.250000000000000]]
```
### Reference elements
Reference elements can be obtained from a finite element:

```python
import symfem

lagrange = symfem.create_element("triangle", "Lagrange", 1)
reference = lagrange.reference
```

Alternatively, reference elements can be created using the `create_reference` function.
For example:

```python
import symfem

triangle = symfem.create_reference("triangle")
interval = symfem.create_reference("interval")
tetrahedron = symfem.create_reference("tetrahedron")
```

Various information about the reference can be accessed:
```python
import symfem

triangle = symfem.create_reference("triangle")
print(triangle.vertices)
print(triangle.edges)
```
```
((0, 0), (1, 0), (0, 1))
((1, 2), (0, 2), (0, 1))
```

```python
import symfem

triangle = symfem.create_reference("triangle")
print(triangle.origin)
print(triangle.axes)
```
```
(0, 0)
((1, 0), (0, 1))
```

```python
import symfem

triangle = symfem.create_reference("triangle")
print(triangle.tdim)
print(triangle.gdim)
```
```
2
2
```

```python
import symfem

triangle = symfem.create_reference("triangle")
print(triangle.sub_entity_types)
```
```
['point', <class 'symfem.core.references.Interval'>, <class 'symfem.core.references.Triangle'>, None]
```
