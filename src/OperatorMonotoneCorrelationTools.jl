module OperatorMonotoneCorrelationTools

# Write your package code here.
using LinearAlgebra

export partialtrace, basischange, returntocompunitary
export choitokraus, krausaction
export Haarrandomunitary, hsrandomstate
export genGellMann, gencompbasis, genNormDiscWeyl
include("basicfunctions.jl")

export perspective, innerproductf, Jfpsigma, getONB
export SchReversalMap, getcontractioncoeff
export Jfpsigmachoi, qmaxcorrcoeff, qmaxlincorrcoeff
include("joperatorfunctions.jl")

end
