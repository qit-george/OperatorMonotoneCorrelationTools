using LinearAlgebra, Test
using OperatorMonotoneCorrelationTools

function fGM(x)
    return sqrt(x)
end
fGM0 = 0
fGMpinf = 0

for λ = 0:0.1:1
    choi = WernerHolevochoi(d, λ)
    Ak, Bk = choitokraus(choi, d, d)
    ρA = hsrandomstate(d,d,true)
    test = qmaxcorrcoeff(ρA, Ak, Bk, fGM, fGM0, fGMpinf)
    println(λ," ",test)
end

#This works, but is slow
d = 3
for λ = 0:0.1:1
    choi = WernerHolevochoi(d, λ)
    Ak, Bk = choitokraus(choi, d, d)
    ρA = 1/d*Matrix(1I,d,d)
    test = qmaxcorrcoeff(ρA, Ak, Bk, fGM, fGM0, fGMpinf)

    ρA2 = kron(ρA,ρA)
    Ak2, Bk2 = _parallelchan(Ak,Bk)
    test2 = qmaxcorrcoeff(ρA2, Ak2, Bk2, fGM, fGM0, fGMpinf)
    println(λ," ",abs(test2-test))
end

#= # This doesn't
d = 2
λ = 0.2
    choi = WernerHolevochoi(d, λ)
    Ak, Bk = choitokraus(choi, d, d)
    ρA = hsrandomstate(d,d,true)
    test = qmaxcorrcoeff(ρA, Ak, Bk, fGM, fGM0, fGMpinf)

    ρA2 = kron(ρA,ρA)
    Ak2, Bk2 = _parallelchan(Ak,Bk)
    test2 = qmaxcorrcoeff(ρA2, Ak2, Bk2, fGM, fGM0, fGMpinf)
    println(λ," ",minimum(eigvals(ρA))," ",abs(test2-test)) =#

function fAM(x)
    return (x+1)/2
end
fAM0 = 1/2
fAMpinf = 0

d = 3
for λ = 0:0.1:1
    choi = WernerHolevochoi(d, λ)
    Ak, Bk = choitokraus(choi, d, d)
    ρA = 1/d*Matrix(1I,d,d)
    testAM = qmaxcorrcoeff(ρA, Ak, Bk, fAM, fAM0, fAMpinf)
    testGM = qmaxcorrcoeff(ρA, Ak, Bk, fGM, fGM0, fGMpinf)
    println("AM: ",testAM," GM: ",testGM)

    ρA2 = kron(ρA,ρA)
    Ak2, Bk2 = _parallelchan(Ak,Bk)
    testAM2 = qmaxcorrcoeff(ρA2, Ak2, Bk2, fGM, fGM0, fGMpinf)
    println(λ," ",abs(testAM2-testAM))
end
