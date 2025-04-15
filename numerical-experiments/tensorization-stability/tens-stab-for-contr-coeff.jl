# This script is for studying the numerical stability of the package for 
# calculating η_{χ^{2}_{f}}. We show it isn't generically numerically stable
# as one might expect. 

using LinearAlgebra, Test
using OperatorMonotoneCorrelationTools

println("This script is for investigating the numerical stability of the
function that computes the contraction coefficient in this package.")

println("To do this, we use that for fGM, the value tensorizes, so we check how the 
value varies between the single copy and double copy case.")
    
dA = 2
dB = 2
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

println("The following two tests show our code works for the qubit depolarizing channel--- 
even with random, complex-valued states.")
println("It also shows that η_{χ^2_{LM}}(ℰ,σ) tensorizes for the depolarizing channel")
@testset "Stability of qubit depolarizing" begin
    println("First we check for f being the geometric mean")
    maxdev = 0
    for q = 0:0.05:1
        println("Now on depolarizing parameter q=", q)
        Ak, Bk = _depolkraus(q)

        for run = 1:10
            σ = hsrandomstate(dA)
            v1 = getcontractioncoeff(Ak, Bk, σ, fGM, f0, fpinf)

            σ2 = kron(σ, σ)
            Ak2, Bk2 = _parallelchan(Ak,Bk)

            v2 = getcontractioncoeff(Ak2, Bk2, σ2, fGM, f0, fpinf)

            dev = abs(v2 - v1)
            dev > maxdev ? maxdev = dev : nothing
        end
    end
    @test maxdev < 1e-10

    println("Next we check for f being the log mean")
    maxdev = 0
    for q = 0:0.05:1
        println("Now on q=", q)
        Ak, Bk = _depolkraus(q)

        for run = 1:10
            #σ = hsrandomstate(dA, dA, true)
            σ = hsrandomstate(dA)
            v1 = getcontractioncoeff(Ak, Bk, σ, fLM, f0, fpinf)

            σ2 = kron(σ, σ)
            Ak2, Bk2 = _parallelchan(Ak,Bk)

            v2 = getcontractioncoeff(Ak2, Bk2, σ2, fLM, f0, fpinf)

            dev = abs(v2 - v1)
            dev > maxdev ? maxdev = dev : nothing
        end
    end
    @test maxdev < 1e-10
end

# The following tests show it works fine for a random channel and the max mixed state
println("The following shows that for a random 2->d channel and max mixed state
the numerical stability seems good.")
@testset "qubit-to-qudit & max mixed small error" begin
    dA = 2
    maxdev = 0
    for dB = 2:7
        println("Now starting (dA,dB)=", (dA, dB))
        choi = randomquantumchannel(dA, dB, true)
        isPSD(choi)
        partialtrace(choi, dA, dB, 2)
        Ak, Bk = choitokraus(choi, dA, dB)

        σ = 1 / dA * Matrix(1I, dA, dA)

        v1 = getcontractioncoeff(Ak, Bk, σ, fGM, f0, fpinf)

        σ2 = kron(σ, σ)
        Ak2, Bk2 = _parallelchan(Ak,Bk)

        v2 = getcontractioncoeff(Ak2, Bk2, σ2, fGM, f0, fpinf)

        dev = abs(v2 - v1)
        dev > maxdev ? maxdev = dev : nothing
    end
    @test maxdev < 1e-12
end

println("The following shows that for a random 2->d channel and a random quantum state the
numerical stability is rather bad.")
@testset "qubit-to-qudit & random state large error" begin
    dA = 2
    maxdev = 0
    for dB = 2:7
        println("Now starting (dA,dB)=", (dA, dB))
        choi = randomquantumchannel(dA, dB, true)
        isPSD(choi)
        partialtrace(choi, dA, dB, 2)
        Ak, Bk = choitokraus(choi, dA, dB)

        σ = hsrandomstate(2,2,true)

        v1 = getcontractioncoeff(Ak, Bk, σ, fGM, f0, fpinf)

        σ2 = kron(σ, σ)
        Ak2, Bk2 = _parallelchan(Ak,Bk)

        v2 = getcontractioncoeff(Ak2, Bk2, σ2, fGM, f0, fpinf)

        dev = abs(v2 - v1)
        println("deviation from tensorization:",dev)
        dev > maxdev ? maxdev = dev : nothing
    end
    @test maxdev > 1e-2
end