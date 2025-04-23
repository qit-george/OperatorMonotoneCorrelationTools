"""
    perspective(x,y,f,f0,fpinf)

For a given function f, this computes the perspective function 
```math
    P_{f}(x,y) \\coloneqq \\begin{cases}
        yf(x/y) & x,y > 0  , \\\\
        yf(0^{+}) & x = 0 , \\\\
        xf'(+\\infty) & y = 0 
    \\end{cases}
```
where ``0f(0/0) \\coloneqq 0``, ``0\\cdot \\infty \\coloneqq 0``,
```math
    f(0^{+}) \\coloneqq \\lim_{x \\downarrow 0} f(x)  , \\quad \\text{and} \\quad f'(+\\infty) \\coloneqq \\lim_{x \\to +\\infty} \\frac{f(x)}{x} . 
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
    innerproductf(X,Y,sigma,p,f,f0,fpinf)

This function computes ``\\langle X , Y \\rangle_{\\mathbf{J}^{p}_{f,\\sigma}}``.
Note that it does not check the function ``f`` is an operator monotone function.
"""
function innerproductf(X,Y,sigma,p,f,f0,fpinf)
    #Note that the eigenvectors are ordered according to increasing eigenvalues
    d = size(sigma)[1]
    Z = zeros(Complex,d,d) 
    X = X' #we just overwrite it as we never use X directly
    Xastp = zeros(Complex,d,d)
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

"""
    Jfpsigma(Y,sigma,p,f,f0,fpinf)

This function computes ``\\mathbf{J}_{f,\\sigma}^{p}(Y)``. It expects the input in the 
computational basis and returns the matrix in the computational basis.
"""
function Jfpsigma(Y,σ,p,f,f0,fpinf)
    size(Y) != size(σ) ? throw(ArgumentError("Y and σ aren't the same dimensions")) : nothing
    d = size(σ)[1]
    Yout = zeros(Complex,d,d)
    λ,basis = eigen(σ)

    # There can be numerical error resulting in imaginary parts in eigenvalues
    # The following controls when you want to be warned about this and/or stop for accuracy reasons
    λ = _makereal(λ)
    normdiff = abs(1-sum(λ)) 
    if normdiff > 1e-6 
        println("Difference from normalization:", normdiff)
        throw(ErrorException("Difference from normalization too large")) 
    end

    #This computes it in the basis of σ
    for i = 1:d
        for j = 1:d
            t = perspective(λ[i],λ[j],f,f0,fpinf)
            if p < 0 && t==0 #keeps track of the pseudoinverse aspect
                Yout[i,j] = 0
            else
                Yout[i,j] = t^p *basis[:,i]'*Y*basis[:,j]
            end
        end
    end
    #Now we convert it back to the computational basis
    U = returntocompunitary(σ)
    #B = diagm(collect(1:1:d)) #The scaling is to guarantee we keep the same ordering of the comp basis
    return U*Yout*U' #basischange(Yout,B)
end

"""
    getONB(σ,p,f,f0,fpinf)

This function performs the (modified) Gram Schmidt process for the 
inner product spaces ``\\langle X,Y \\rangle_{\\mathbf{J}_{f,\\sigma}^{p}}``
considered in the paper. 

Note inputs need to be in computational basis and are returned in computational basis 
as the inner product value is a number and thus does not change the basis here.
"""
function getONB(σ,p,f,f0,fpinf)
    #Generate initial ONB
    d = size(σ)[1]
    onb = genGellMann(d)
    pushfirst!(onb, sqrt(σ))

    #Apply modified Gram-Schmidt process
    for i in eachindex(onb)
        onb[i] = 1 / sqrt(innerproductf(onb[i], onb[i], σ, p, f, f0, fpinf)) * onb[i]
        for j = i+1:length(onb)
            onb[j] = onb[j] - innerproductf(onb[i], onb[j], σ, p, f, f0, fpinf) * onb[i]
        end
    end

    return onb
end

"""
    SchReversalMap(X,Ak,Bk,σ,f,f0,fpinf)
   
Applies the Schrodinger reversal map to X according to ``f``,``\\mathcal{E}``, and ``σ.`` 
``\\mathcal{E}`` is presumed to be provided in its Kraus operator form.
It is assumed all inputs are expressed in the computational basis.
"""
function SchReversalMap(X,Ak,Bk,σ,f,f0,fpinf)
    σout = krausaction(Ak,Bk,σ)
    step1 = Jfpsigma(X,σout,-1,f,f0,fpinf) 
    step2 = krausaction(Ak', Bk', step1) #Apply adjoint map
    return Jfpsigma(step2,σ,1,f,f0,fpinf)
end

"""
    getcontractioncoeff(Ak, Bk, σ, f, f0, fpinf)

This returns the contraction coefficient ``\\eta_{\\chi^{2}_{f}}(\\mathcal{E},\\sigma)`` 
for a full rank input state ``\\sigma`` and symmetric-inducing operator monotone function ``f.``
"""
function getcontractioncoeff(Ak, Bk, σ, f, f0, fpinf)
    d = size(σ)[1]
    rank(σ) != d ? throw(ArgumentError("σ must be full rank")) : nothing

    onb = getONB(σ, -1, f, f0, fpinf)

    T = zeros(Complex, d^2, d^2)
    for j = 1:d^2
        #Action of 𝒮_{f,ℰ,σ}∘ℰ on e_{j}
        ejout = krausaction(Ak, Bk, onb[j])
        ejout = SchReversalMap(ejout, Ak, Bk, σ, f, f0, fpinf)
        for i = 1:d^2
            T[i, j] = innerproductf(onb[i], ejout, σ, -1, f, f0, fpinf)
        end
    end

    #return T
    λ = eigvals(T)
    return λ[d^2-1] #Eigvals returns the eigenvalues in increasing order
end

"""
    Jfpsigmachoi(σ,p,f,f0,fpinf)

This function returns the Choi operator of ``\\mathbf{J}_{f,\\sigma}^{p}.``
We note this has a specific function for obtaining the Choi operator
to force the user to consider ``p,f,f(0+),f'(+\\inf).``
"""
function Jfpsigmachoi(σ,p,f,f0,fpinf)
    d = size(σ)[1]
    choimat = zeros(d^2,d^2)
    for i = 1:d
        for j = 1:d
            elmat = zeros(d,d)
            elmat[i,j] = 1
            choimat = choimat + kron(elmat,Jfpsigma(elmat,σ,p,f,f0,fpinf))
        end
    end
    return choimat
end


"""
    qmaxcorrcoeff(ρA::Matrix, Ak::Vector, Bk::Vector, f, f0, fpinf)

This function computes the maximal correlation coefficient ``\\mu_{f}(\\rho_{AB})``
when given ``\\rho_{A}`` and the Kraus operators of ``\\mathcal{E}`` such that 
``\\rho_{AB} = (\\text{id}_{A} \\otimes \\mathcal{E})(\\psi_{\\rho_{A}})`` where
``\\psi_{\\rho_{A}}`` is the canonical purification of ``\\rho_{A}``.
"""
function qmaxcorrcoeff(ρA::Matrix, Ak::Vector, Bk::Vector, f, f0, fpinf)
    #preliminary calculations
    dA = size(ρA)[1]
    ρAsq = sqrt(ρA)
    ρB = krausaction(Ak, Bk, ρA)
    dB = size(ρB)[1]

    #Kraus of Jf maps
    ΩJfρAneghalf = Jfpsigmachoi(ρA, -1 / 2, f, f0, fpinf)
    C, D = choitokraus(ΩJfρAneghalf, dA, dA)
    ΩJfρBneghalf = Jfpsigmachoi(ρB, -1 / 2, f, f0, fpinf)
    E, F = choitokraus(ΩJfρBneghalf, dB, dB)

    #Kraus of Λ_{ ̃ρ_f}* ∘ Λ_{ ̃ρ_f}
    Mw = Matrix{Any}[]
    Rw = Matrix{Any}[]
    for x1 in eachindex(Ak)
        for x2 in eachindex(Ak)
            for y1 in eachindex(C)
                for y2 in eachindex(C)
                    for z1 in eachindex(E)
                        for z2 in eachindex(E)
                            push!(Mw, conj.(C[y2]) * ρAsq * Ak[x2]' * E[z2]' * E[z1] * Ak[x1] * transpose(ρAsq) * transpose(C[y1]))
                            push!(Rw, F[z2]' * Bk[x2]' * ρAsq * conj.(D[y2]) * transpose(D[y1]) * transpose(ρAsq) * Bk[x1] * F[z1])
                        end
                    end
                end
            end
        end
    end

    #Get ONB of Herm(A) with respect to HS
    onb = genGellMann(dA)
    pushfirst!(onb, 1/sqrt(dA)*Matrix(1I,dA,dA))

    #Get standard matrix T
    T = zeros(Complex, dA^2, dA^2)

    for j in 1:dA^2
        #Action of Λ_{ ̃ρ_f}* ∘ Λ_{ ̃ρ_f} on ONB
        ejout = krausaction(Mw, Rw, onb[j])
        for i in 1:dA^2
            T[i, j] = tr(onb[i]' * ejout)
        end
    end

    #Get the eigenvalues
    λ = eigvals(T)

    # There can be numerical error resulting in imaginary parts in eigenvalues
    λ = _makereal(λ)

    val = sqrt(λ[dA^2-1])

    #There can be numerical error, so we process this a little bit if it exceeds unity
    val > 1 + 1e-8 ? throw(ErrorException("The value is greater than 1+1e-8, so the numerical error is bad")) : nothing
    if 1 + 1e-8 > val > 1
        @warn("the value is slightly above unity, there is some numerical error")
        val = 1
    end

    return val
end

"""
    qmaxlincorrcoeff(ρA::Matrix, Ak::Vector, Bk::Vector,k)

This function computes the maximal correlation coefficient ``\\mu_{f_{k}}(\\rho_{AB})``
for ``f_{k}(x) = x^{k}.`` Currently it requires that it is given ``\\rho_{A}`` and 
the Kraus operators of ``\\mathcal{E}`` such that 
``\\rho_{AB} = (\\text{id}_{A} \\otimes \\mathcal{E})(\\psi_{\\rho_{A}})`` where
``\\psi_{\\rho_{A}}`` is the canonical purification of ``\\rho_{A}``.
"""
function qmaxlincorrcoeff(ρA::Matrix, Ak::Vector, Bk::Vector, k)
    k<0 || k>1 ? Throw(ArgumentError("k must be between 0 and 1")) : nothing
    #preliminary calculations
    dA = size(ρA)[1]
    ρAsq = sqrt(ρA)
    ρATsq = sqrt(transpose(ρA))
    ρAbar = conj.(ρA)
    ρAbark = (ρAbar)^(-k / 2)
    ρAbarkp = (ρAbar)^(-(1 - k) / 2)

    ρB = krausaction(Ak, Bk, ρA)
    ρBk = ρB^(-k / 2)
    ρBkp = ρB^(-(1 - k) / 2)

    #Kraus of Λ_{ ̃ρ_k}
    Gk = Matrix{Any}[]
    Lk = Matrix{Any}[]
    for k in eachindex(Ak)
        push!(Gk, ρBk * Ak[k] * ρATsq * ρAbarkp)
        push!(Lk, ρBkp * Bk[k] * ρATsq' * ρAbark')
    end

    #Kraus of  Λ*_{ ̃ρ_k} ∘ Λ_{ ̃ρ_k}
    Mw = Matrix{Any}[]
    Rw = Matrix{Any}[]
    for k1 in eachindex(Ak)
        for k2 in eachindex(Ak)
            push!(Mw, Gk[k2]' * Gk[k1])
            push!(Rw, Lk[k2]' * Lk[k1])
        end
    end

    #Get ONB of Herm(A) with respect to HS
    onb = genGellMann(dA)
    pushfirst!(onb, 1/sqrt(dA)*Matrix(1I,dA,dA))

    #Get standard matrix T
    T = zeros(Complex, dA^2, dA^2)

    for j in 1:dA^2
        #Action of Λ_{ ̃ρ_f}* ∘ Λ_{ ̃ρ_f} on ONB
        ejout = krausaction(Mw, Rw, onb[j])
        for i in 1:dA^2
            T[i, j] = tr(onb[i]' * ejout)
        end
    end

    #Get the eigenvalues
    λ = eigvals(T)

    # There can be numerical error resulting in imaginary parts in eigenvalues
    λ = _makereal(λ)
    
    val = sqrt(λ[dA^2-1])

    #There can be numerical error, so we process this a little bit if it exceeds unity
    val > 1 + 1e-8 ? throw(ErrorException("The value is greater than 1+1e-8, so the numerical error is bad")) : nothing
    if 1 + 1e-8 > val > 1
        @warn("the value is slightly above unity, there is some numerical error")
        val = 1
    end

    return val
end