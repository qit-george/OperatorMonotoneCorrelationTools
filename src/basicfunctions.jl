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
The identity relies on the vec mapping in the computational bases: ``vec:\\vert j \\rangle_{B} \\langle i \\vert_{A} \\to \\langle i \\vert_{A}vec:\\vert j \\langle_{B}``.
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

#= function choitokraus(choi,dA,dB)
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
end =#

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
    hsrandomstate(d::Int,k::Int=d)

Draws a density matrix according to the ``\\mu_{nk}`` 
distribution. The method of construction follows Lemma 1
of "Asymptotics of random density matrices" by Ion Nechita
"""
function hsrandomstate(d::Int,k::Int=d)
    #Construct Z that is made up of iid 𝒩_{ℂ}(0,1)
    #which means Z[i,j] = X[i,j] +iY[i,j] where X,Y ∼ 𝒩(0,1/2)
    #thus we need to rescale randn by 1/2
    Z= sqrt(1/2)*randn(d,k) + sqrt(1/2)*1im*randn(d,k)
    return( Z*Z'/tr(Z*Z'))
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

"""
    gencompbasis(d)

This function returns the computational basis of dimension d.
"""
function gencompbasis(d)
    basis = Vector{Any}[]
    init = zeros(d)
    for i = 1:d
        init[i] = 1
        push!(basis,init)
        init[i] = 0
    end
    return basis
end

"""
    genNormDiscWeyl(d)

This function returns the normalized discrete Weyl operators for dimension d.
"""
function genNormDiscWeyl(d)
    ζ = exp(2 * π * 1im / d)
    X = zeros(d, d)
    #Define the "Clock Operator"
    X[1, d] = 1
    for i = 1:d-1
        X[i+1, i] = 1
    end
    #Define the "Phase Operator"
    Z = zeros(Complex, d, d)
    for i = 1:d
        Z[i, i] = ζ^(i - 1)
    end
    Z
    Wab = Matrix{Complex}[]
    for i = 0:d-1
        for j = 0:d-1
            push!(Wab, 1 / sqrt(d) * X^(i) * Z^(j))
        end
    end
    return Wab
end