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
    innerproductf(X,Y,sigma,p,f,f0,fpinf)

This function computes ``\\langle X , Y \\rangle_{\\mathbf{J}^{p}_{f,\\sigma}}``.
Note that it does not check the function f is an operator monotone function.
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
    Jfpsigma(Y,sigmap,f,f0,fpinf)

This function computes ``\\mathbf{J}_{f,\\sigma}^{p}(Y)``. It assumes that everything is 
provided in the computational basis and returns it also in the computational basis.
"""
function Jfpsigma(Y,sigma,p,f,f0,fpinf)
    size(Y) != size(sigma) ? throw(ArgumentError("Y and σ aren't the same dimensions")) : nothing
    d = size(sigma)[1]
    Yout = zeros(Complex,d,d)
    λ,basis = eigen(sigma) 
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
    B = diagm(collect(1:1:d)) #The scaling is to guarantee we keep the same ordering of the comp basis
    return basischange(Yout,B)
end

"""
    getONB(σ,p,f,f0,fpinf)

This function performs the (modified) Gram Schmidt process for the 
inner product spaces ``\\langle X,Y \\rangle_{\\mathbf{J}_{f,\\sigma}^{p}}``
considered in the paper.
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