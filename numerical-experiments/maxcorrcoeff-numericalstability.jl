using LinearAlgebra
using Test
using OperatorMonotoneCorrelationTools

function fGM(x)
    return sqrt(x)
end

function fLM(x)
    if x != 1 && x > 0
        return (x - 1) / log(x)
    elseif x == 1
        return 1
    else
        throw(ArgumentError("x cannot be negative"))
    end
end
f0 = 0
fpinf = 0

function fAM(x)
    return (x+1)/2
end
fAM0 = 1/2
fAMpinf = 1/2


dA = 2
maxdev = 0
for q = 0:0.1:1
    idMat = [1 0; 0 1]
    sigmaX = [0 1; 1 0]
    sigmaY = [0 -1im; 1im 0]
    sigmaZ = [1 0; 0 -1]
    Ak = [sqrt(1 - 3 * q / 4) * idMat, sqrt(q / 4) * sigmaX, sqrt(q / 4) * sigmaY, sqrt(q / 4) * sigmaZ]
    Bk = Ak

    σ = [3/4 1/4 ; 1/4 1/4]
    #σ = hsrandomstate(dA, dA, true)
    v1 = qmaxcorrcoeff(σ, Ak, Bk, fGM, f0, fpinf)

    σ2 = kron(σ, σ)
    Ak2 = Matrix{Any}[]
    Bk2 = Matrix{Any}[]
    for i in eachindex(Ak)
        for j in eachindex(Ak)
            push!(Ak2, kron(Ak[i], Ak[j]))
            push!(Bk2, kron(Bk[i], Bk[j]))
        end
    end

    v2 = qmaxcorrcoeff(σ2, Ak2, Bk2, fGM, f0, fpinf)

    dev = abs(v2 - v1)
    println("q=",q,"dev=",dev)
    dev > maxdev ? maxdev = dev : nothing
end
maxdev


dA = 2
maxdev = 0
for q = 0:0.1:1
    idMat = [1 0; 0 1]
    sigmaX = [0 1; 1 0]
    sigmaY = [0 -1im; 1im 0]
    sigmaZ = [1 0; 0 -1]
    Ak = [sqrt(1 - 3 * q / 4) * idMat, sqrt(q / 4) * sigmaX, sqrt(q / 4) * sigmaY, sqrt(q / 4) * sigmaZ]
    Bk = Ak

    #σ = 1 / 2 * [1 0; 0 1]
    σ = hsrandomstate(dA, dA, true)
    v1 = qmaxcorrcoeff(σ, Ak, Bk, fAM, fAM0, fAMpinf)

    σ2 = kron(σ, σ)
    Ak2 = Matrix{Any}[]
    Bk2 = Matrix{Any}[]
    for i in eachindex(Ak)
        for j in eachindex(Ak)
            push!(Ak2, kron(Ak[i], Ak[j]))
            push!(Bk2, kron(Bk[i], Bk[j]))
        end
    end

    v2 = qmaxcorrcoeff(σ2, Ak2, Bk2, fAM, fAM0, fAMpinf)

    dev = abs(v2 - v1)
    dev > maxdev ? maxdev = dev : nothing
end
maxdev