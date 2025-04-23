"""
    isPSD(X)

Returns true if X is a positive semidefinite operator.
"""
function isPSD(X)
    λ = eigvals(X)
    #The eigenvalues may have small imaginary parts
    λ = _makereal(λ)
    if all(>=(-1e-14), λ)
        return true 
    else 
        return false
    end
end


"""
    partialtrace(ρ,dA,dB,sys)

This function computes the partial trace on a bipartite state ``\\rho_{AB}``.
If sys = 1, it traces over the A system and otherwise the B system.
"""
function partialtrace(ρ,dA,dB,sys)
    if sys == 1
        #This uses that tracing over A sums the diagonal blocks
        marg = zeros(Complex,dB,dB)
        for i = 1:dA
            marg = marg + ρ[(i-1)*dB+1:i*dB,(i-1)*dB+1:i*dB]
        end
    else
        #This uses that tracing over B results in each entry being
        #the trace of the corresponding subblock
        marg = zeros(Complex,dA,dA)
        for i in 1:dA
            for j in 1:dA  
                marg[i,j] = tr(ρ[(i-1)*dB+1:i*dB,(j-1)*dB+1:j*dB])
            end
        end
    end
    return marg
    #TODO probably the right way to write this generally is to 
    #permute the systems to trace over to act as A and
    #the rest to act as B. Then just use that partial trace
    #and permute the remaining systems back
end

"""
    basischange(A,B)

Expresses a square linear operator A in the eigenbasis of B where
the eigenbasis is expressed with the k-th eigenvector corresponding to
the k-th largest eigenvalue of B.
"""
function basischange(A,B)
    !isapprox(B,B',atol=1e-6) ? throw(ArgumentError("B is not hermitian")) : nothing
    size(A)[1] != size(A)[2] ? throw(ArgumentError("A is not square")) : nothing
    d = size(B)[1]
    basis = eigvecs(B)
    Ap = zeros(Complex,d,d)
    for i = 1:size(B)[1]
        for j = 1:size(B)[2]
            Ap[i,j] = basis[:,i]'*A*basis[:,j]
        end
    end
    return Ap
end

"""
    returntocompunitary(A)

Given A in the computational basis, this returns the unitary U
such that it takes the eigenbasis of A to the computational basis.
This is for taking matrices written in the eigenbasis of A back to
the computational basis.
"""
function returntocompunitary(A)
    d = size(A)[1]
    λ1, initbasis = eigen(A)
    finalbasis = gencompbasis(d)
    U = zeros(Complex, d, d)
    for j = 1:d
        for i = 1:d
            U[i, j] = finalbasis[i]'*initbasis[:,j]
        end
    end
    return U
end

"""
    choitokraus(choi,dA,dB)

Converts a Choi operator of a linear map to its Kraus representation.
The identity relies on the vec mapping in the computational bases: ``vec:\\vert j \\rangle_{B} \\langle i \\vert_{A} \\to \\langle i \\vert_{A} \\langle j \\vert _{B}``.
This is equivalent to stacking columns of the matrix on top of each other, which is the vec mapping for Julia.
"""
function choitokraus(choi, dA, dB)
    r = rank(choi)
    dAB = dA * dB
    Ak = Matrix{Any}[]
    Bk = Matrix{Any}[]
    isHP = false
    norm(choi - choi') < 1e-12 ? isHP=true : nothing #Checks if it is a Hermitian preserving map
    if isHP #At some point, one could generalize to Hermitian preserving maps
        λ, vecs = eigen(choi)
        λ = _makereal(λ)
        if all(>=(-1e-14), λ) #simplified for CP maps
            for i = 0:rank(choi)-1
                push!(Ak, sqrt(λ[dAB-i]) * reshape(vecs[:, dAB-i], dB, dA))
                push!(Bk, sqrt(λ[dAB-i]) * reshape(vecs[:, dAB-i], dB, dA))
            end
        else #For general linear maps, need to use the SVD
            F = svd(choi)
            for i = 1:r
                push!(Ak, sqrt(F.S[i]) * reshape(F.U[:, i], dB, dA))
                push!(Bk, sqrt(F.S[i]) * reshape(F.V[:, i], dB, dA))
            end
        end
    end

    return Ak, Bk
end

"""
    krausaction(Ak,Bk,input)

Implements the action of a linear map ``\\Phi_{A \\to B}`` given its Kraus operators ``\\{A_{k},B_{k}\\}_{k}``,
i.e.  ``\\Phi(X) = \\sum_{k} A_{k}XB_{k}^{\\ast}``.
"""
function krausaction(Ak,Bk,input)
    #Sanity checks
    length(Ak) != length(Bk) ? ArgumentError("Left and right Kraus operators not the same length") : nothing
    #This could be iterated over the whole list for a complete check
    size(Ak[1]) != size(Bk[1]) ? ArgumentError("Left and right Kraus operators don't map between the same size spaces") : nothing

    dout = size(Ak[1])[1]
    din = size(Ak[1])[2]
    rhoout = zeros(Complex,dout,dout)
    for i = 1:length(Ak)
        rhoout = rhoout + Ak[i]*input*Bk[i]'
    end

    return rhoout
end

"""
    swapoperator(d)

Returns the swap operator ``\\mathbb{F} = \\sum_{i,j \\in [d]} \\vert i \\langle \\rangle j \\vert \\otimes \\vert j \\langle \\rangle i \\vert ,``
where ``d`` denotes the dimension.
"""
function swapoperator(d :: Integer)
    𝔽 = zeros(d^2,d^2)
    Eij = zeros(d, d)
    Eji = zeros(d,d)
    for i = 1:d
        for j = 1:d
            Eij[i,j] = 1
            Eji[j,i] = 1
            𝔽 = 𝔽 + kron(Eij, Eji)
            Eij[i,j] = 0
            Eji[j,i] = 0
        end
    end
    return 𝔽
end

"""
    WernerHolevochoi(d,q)
Returns the Choi operator of the Werner-Holevo channel  with
parameter q acting on a d-dimensional space: 
``\\mathcal{W}_{q} := (1-q) \\mathcal{W}_{sym} + q \\mathcal{W}_{as}.``
"""
function WernerHolevochoi(d,q)
    id = Matrix(1I,d^2,d^2)
    𝔽 = swapoperator(d)
    choiWH0 = 1/(d + 1) * (id + 𝔽)
    choiWH1 = 1/(d - 1) * (id - 𝔽)
    choi = (1-q)*choiWH0 + q*choiWH1
    return choi
end
