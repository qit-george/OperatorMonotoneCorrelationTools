module OperatorMonotoneCorrelationTools

# Write your package code here.
using LinearAlgebra

export partialtrace, basischange, returntocompunitary
export choitokraus, krausaction, isPSD, swapoperator
include("basicfunctions.jl")

export Haarrandomunitary, hsrandomstate, randomquantumchannel
include("randomobjects.jl")

export genGellMann, gencompbasis, genNormDiscWeyl
include("commonsetgenerators.jl")

export perspective, innerproductf, Jfpsigma, getONB
export SchReversalMap, getcontractioncoeff
export Jfpsigmachoi, qmaxcorrcoeff, qmaxlincorrcoeff
include("joperatorfunctions.jl")

export _makereal, _parallelchan, _depolkraus
include("helperfunctions.jl")

end
