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

The finite element method (FEM) is a popular method for numerically solving a wide
range of partial differential equations (PDEs). To solve a problem using FEM, the PDE is first
written in a weak form, for example: find $u\in V$ such that for all $v\in V,$

$$\int_\Omega \nabla u\cdot\nabla v=\int_\Omega fv,$$

where $f$ is a known function, and $\Omega$ is the domain on which the problem is begin solved.
This form is then discretised by defining a finite dimensional subspace of $V$---often called
$V_h$---and looking for a solution $u_h\in V_h$ that satisfies the above equation for all functions
$v_h\in V_h$. These finite dimensional subspaces are defined by meshing the domain of the problem,
then defining a set of basis functions on each cell in the mesh (and enforcing any desired
continuity between the cells).

For different applications, there are a wide range of finite dimensional spaces that can be used.
Symfem is a Python library that can be used to symbolically compute basis functions of these
spaces. The symbolic representations are created using Sympy [@sympy], allowing
them to be easily manipulated using Sympy's functionality once they are created.

# Statement of need

In FEM libraries, it is common to define basis functions so that they, and their
derivatives, can quickly and efficiently be evaluated at a collection of points, thereby allowing
full computations to be completed quickyl. The libraries FIAT [@fiat] and Basix [@basix]---which
are part of the FEniCS project [@fenics]---implement this functionality as stand-alone libraries.
Many other FEM libraries define their basis functions as part of the core library functionality.
It is not common to be able to compute a symbolic representation of the basis functions.

Symfem offers a wider range of finite element spaces than other FEM libraries, and the ability
to symbolically compute basis functions. There are a number of situations in which the symbolic
representation of a basis function is useful: it is easy to confirm, for example, that the
derivatives of the basis functions have a certain desired property, or check what they are
equal to when restricted to one face or edge of the cell.

Symfem can also be used to explore the behaviour of the wide range of spaces it supports, so the
user can decide which spaces to implement in a faster way in their FEM code. Additionally,
Symfem can be used to prototype new finite element spaces, as custom spaces can easily be
added, then it can be checked that the basis functions of the space behave as expected.

As basis functions are computed symbolically in Symfem, it is much slower than the alternative
libraries. It is therefore not suitable for performing actual finite element calculations. It
should instead be seen as a library for research and experimentation.

# References
