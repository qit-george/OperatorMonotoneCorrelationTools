"""
    choitokraus(choi)

Converts a Choi operator of a linear map to its Kraus representation.
The identity relies on the vec mapping in the computational bases: ``vec:\\vert j \\rangle_{B} \\langle i \\vert_{A} \\to \\langle i \\vert_{A}vec:\\vert j \\langle_{B}``.
This is equivalent to stacking columns of the matrix on top of each other, which is the vec mapping for Julia.
"""
function choitokraus(choi,dA,dB)
    r = rank(choi)
    F = svd(choi)
    #One may verify that reshape acts like the inverse of the vec mapping
    
    Ak = Matrix{Any}[]
    Bk = Matrix{Any}[]
    for i = 1:rank(choi)
        push!(Ak, reshape(sqrt(F.S[i])*F.U[:,i], (dA,dB)))
        push!(Bk, reshape(sqrt(F.S[i])*F.Vt'[:,i], (dA,dB)))
    end

    return Ak,Bk
end

function convertAtobasisofB(A,B)
    
end

function operatorJ(f,sigma)
end