"""
    choitokraus(choi,dA,dB)

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

"""
    krausaction(Ak,Bk,input)

Implements the action of a linear map given its kraus operators.
"""
function krausaction(Ak,Bk,input)
    #Sanity checks
    length(Ak) != length(Bk) ? ArgumentError("Left and right Kraus operators not the same length") : nothing
    #This could be iterated over the whole list for a complete check
    size(Ak[1]) != size(Bk[1]) ? ArgumentError("Left and right Kraus operators don't map between the same size spaces") : nothing

    dout = size(Ak[1])[1]
    din = size(Ak[1])[2]
    rhoout = zeros(dout,dout)
    for i = 1:length(Ak)
        rhoout = rhoout + Ak[i]*input*Bk[i]'
    end

    return rhoout
end

"""
    basischange(A,B)

Expresses a square linear operator A in the eigenbasis of B where
the eigenbasis is expressed with the k-th eigenvector corresponding to
the k-th largest eigenvalue of sigma.
"""
function basischange(A,B)
    !isapprox(B,B',atol=1e-6) ? throw(ArgumentError("B is not hermitian")) : nothing
    size(A)[1] != size(A)[2] ? throw(ArgumentError("A is not square")) : nothing
    d = size(B)[1]
    basis = eigvecs(B)
    Ap = zeros(d,d)
    for i = 1:size(B)[1]
        for j = 1:size(B)[2]
            Ap[i,j] = basis[:,i]'*A*basis[:,j]
        end
    end
    return Ap
end