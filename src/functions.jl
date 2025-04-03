"""
    choitokraus(choi,dA,dB)


    choitokraus(choi,dA,dB)

Converts a Choi operator of a linear map to its Kraus representation.
The identity relies on the vec mapping in the computational bases: ``vec:\vert j \rangle_{B} \langle i \vert_{A} \to \langle i \vert_{A}vec:\vert j \langle_{B}``.
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
    perspective(x,y,f,f0,fpinf)

For a given function f, this computes the perspective function 
```math
    P_{f}(x,y) \\coloneq \\begin{cases}
        yf(x/y) & x,y > 0  , \\
        yf(0^{+}) & x = 0 , \\
        xf'(+\\infty) & y = 0 
    \\end{cases}
```
where ``0f(0/0) \\coloneq 0``, ``0\\cdot \\infty \\coloneq 0``,
```math
    f(0^{+}) \\coloneq \\lim_{x \\downarrow 0} f(x)  , \\quad \\text{and} \\quad f'(+\\infty) \\coloneq \\lim_{x \\to +\\infty} \\frac{f(x)}{x} . 
```
We note that we allow one to control f0 and fpinf. The function will not work if these values are wrong. 
We assume you will put Inf (resp. -Inf) if f0 or fpinf is infinite.
"""
function perspective(x,y,f,f0,fpinf)
    if x > 0 && y > 0
        return y*f(x/y)
    elseif x ==0 && y > 0
        if f0 == Inf || f0 == -Inf
            throw(ArgumentError("Perspective function is infinite"))
        else
            return y*f0
        end
    elseif x > 0 && y == 0
        if fpinf == Inf || fpinf == -Inf
            throw(ArgumentError("Perspective function is infinite"))
        else
            return x*fpinf
        end
    elseif x == 0 && y == 0
        return 0
    else
        throw(ArgumentError("neither x nor y can be negative"))
    end
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

"""
    innerproductf(X,Y,sigma,p,f,f0,fpinf)

This function computes ``\\langle X , Y \\rangle_{\\mathbf{J}^{p}_{f,\\sigma}}``.
Note that it does not check the function f is an operator monotone function.
"""
function innerproductf(X,Y,sigma,p,f,f0,fpinf)
    #Note that the eigenvectors are ordered according to increasing eigenvalues
    d = size(sigma)[1]
    Z = zeros(d,d)
    X = X' #we just overwrite it as we never use X directly
    Xastp = zeros(d,d)
    λ,basis = eigen(sigma)
    for i = 1:d
        for j = 1:d
            Xastp[i,j] = basis[:,i]'*X*basis[:,j]
            t = perspective(λ[i],λ[j],f,f0,fpinf)
            if p < 0 && t==0 #keeps track of the pseudoinverse aspect
                Z[i,j] = 0
            else
                Z[i,j] = t^p*basis[:,i]'*Y*basis[:,j]
            end
        end
    end
    return tr(Xastp*Z)
end