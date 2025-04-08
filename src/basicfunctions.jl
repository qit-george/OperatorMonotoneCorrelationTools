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
the k-th largest eigenvalue of sigma.
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
    rhoout = zeros(Complex,dout,dout)
    for i = 1:length(Ak)
        rhoout = rhoout + Ak[i]*input*Bk[i]'
    end

    return rhoout
end

"""
    RandomUnitary(d)

This function returns a unitary of dimension d according to the Haar measure.
The construction follows "How to generate a random unitary matrix" by
Maris Ozols.
"""
function Haarrandomunitary(d)
    Z = randn(d,d) + 1im*randn(d,d)
    Q,R = qr(Z)
    Λ = diagm(sign.(diag(R)))
    return Q*Λ
end

"""
    genGellMan(d)

This function constructs and returns the generalized Gell Mann matrices.
"""
function genGellMann(d)
    genGMmats = Matrix{Complex}[]
    for n = 1:d
        for m = 1:d
            if n==d && m == d
                nothing
            elseif n == m && n != d
                Gnn = zeros(d,d)
                for i = 1:n
                    Gnn[i,i] = 1
                end
                Gnn[n+1,n+1] = -n
                Gnn = 1/sqrt(n*(n+1))*Gnn
                push!(genGMmats,Gnn)
            elseif n < m
                Enm = zeros(d,d)
                Enm[n,m] = 1
                push!(genGMmats,1/sqrt(2)*(Enm + Enm'))
            else
                Enm = zeros(d,d)
                Enm[n,m] = 1
                push!(genGMmats,1im/sqrt(2)*(Enm - Enm'))
            end
        end
    end
    return genGMmats
end