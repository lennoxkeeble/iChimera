# script to install all the necessary dependencies
using Pkg

dependencies = [
    "StaticArrays",
    "QuadGK",
    "Elliptic",
    "ForwardDiff",
    "FiniteDifferences",
    "BSplineKit",
    "Symbolics",
    "DifferentialEquations",
    "Dierckx",
    "LaTeXStrings",
    "Plots",
    "PlotlyJS",
    "ArbNumerics",
    "HCubature",
    "SpecialFunctions",
    "DelimitedFiles",
    "Combinatorics",
    "Roots",
    "LinearAlgebra"
]
Pkg.add(dependencies)