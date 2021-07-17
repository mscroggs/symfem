---
title: 'Symfem: a symbolic finite element definition library'
tags:
  - Python
  - finite element method
  - basis functions
  - symbolic algebra
  - numerical analysis
authors:
  - name: Matthew W. Scroggs
    orcid: 0000-0002-4658-2443
    affiliation: 1
affiliations:
 - name: Department of Engineering, University of Cambridge
   index: 1
date: 15 July 2021
bibliography: paper.bib
---

# Summary

The finite element method (FEM) is a popular method that is used to numerically solving a wide
range of partial differential equations (PDEs). To solve a problem using FEM, the PDE is first
written in a weak form, for example: find $u\in V$ such that for all $v\in V,$

$$\int_\Omega \nabla u\cdot\nabla v=\int_\Omega fv,$$

where $f$ is a known function, and $\Omega$ is the domain on which the problem is defined.
This form is then discretised by defining a finite dimensional subspace of $V$, and looking for a
solution in this subspace that satisfied the equation for all functions $v$ in the subspace. These
finite dimensional subspaces are defined by meshing the domain of the problem, then defining
a set of basis function on each cell in the mesh (and enforcing any desired continuity between the
cells).

For different applications, there are a wide range of finite dimensional spaces that can be used.
Symfem is a Python library that can be used to compute symbolic representations of the basis
functions of these spaces. The symbolic representation are created using Sympy [@sympy], allowing
them to be easily manipulated using Sympy's functionality once they are created.

# Statement of need

In finite element libraries, it is common to define basis functions so that they, and their
derivatives, can quickly and efficiently be evaluated at a collection of points. The libraries
FIAT [@fiat] and Basix [@basix]---which are part of the FEniCS project [@fenics]---implement
this functionality as stand-alone libraries. Many other finite element libraries define their
basis functions as part of the core library functionality.

As is computes its basis functions symbolically, Symfem is slower that these libraries (and
therefore higher order elements are often impractical to compute). In some situations, however,
it is useful to have a symbolic representation of the basis functions of a space.

When validating
useful for:

- prototyping
- validation
    
# References
