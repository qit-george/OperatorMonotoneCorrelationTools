module OperatorMonotoneCorrelationTools

# Write your package code here.
using LinearAlgebra

export partialtrace, basischange, returntocompunitary
export choitokraus, krausaction
export Haarrandomunitary, hsrandomstate
export genGellMann, gencompbasis
include("basicfunctions.jl")

export perspective, innerproductf, Jfpsigma, getONB
export SchReversalMap, getcontractioncoeff
export Jfpsigmachoi, qmaxcorrcoeff
include("joperatorfunctions.jl")

end
