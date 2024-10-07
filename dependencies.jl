# script to install all the necessary dependencies
using Pkg

dependencies = [
    "StaticArrays",
    "DifferentialEquations",
    "HDF5",
    "LinearAlgebra",
    "Elliptic",
    "QuadGK",
    "PolynomialRoots",
    "GSL",
    "Roots",
    "Combinatorics",
    "JLD2",
    "Printf",
    "LaTeXStrings",
    "Plots",
    "BenchmarkTools",
    "Interpolations",
    "IJulia",
    ]
    
Pkg.add(dependencies)