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

"""
    genGellMan(d)

This function constructs and returns the generalized Gell Mann matrices for dimension d.
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