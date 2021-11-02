"""Symfem: a symbolic finite element definition library."""
from .create import add_element, create_reference, create_element
from .version import version as __version__

__citation__ = """@article{symfem,
  AUTHOR = {Scroggs, Matthew W.},
   TITLE = {Symfem: a symbolic finite element definition library},
 JOURNAL = {Journal of Open Source Software},
    YEAR = {2021},
  VOLUME = {6},
  NUMBER = {64},
   PAGES = {3556},
     DOI = {10.21105/joss.03556},
}"""
