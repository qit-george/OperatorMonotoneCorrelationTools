# This script is for investigating the gap between μ_{f_{k,sym}}(ρAB) and μ^{Lin}_{f_{k}}(ρAB)
# as was observed for one case in "A New Quantum Data Processing Inequality" by Salman Beigi

using LinearAlgebra, Test
using OperatorMonotoneCorrelationTools

function fksym(x,k)
    return 1/2*(x^(k)+x^(1-k))
end

totalruns = 0
gappedruns = 0
for k = 0.6:0.1:0.9
    for q = 0.1:0.1:0.5
        Ak, Bk = _depolkraus(q)
        for run = 1:10
            ρA = hsrandomstate(2, 2, true)
            v1 = qmaxcorrcoeff(ρA, Ak, Bk, x -> fksym(x, k), 0, 0)
            v2 = qmaxlincorrcoeff(ρA, Ak, Bk, k)
            if v2 - v1 > 1e-1
                gappedruns = gappedruns + 1
            elseif v2 < v1
                println("here")
            end
            totalruns = totalruns + 1
        end
    end
end