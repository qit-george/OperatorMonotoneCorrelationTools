"""
    Eigenvalues need to be real, but numerical methods
    sometimes have some small error. This checks if there 
    will be an issue and if there won't be, it returns the
    vector with real elements.
"""
function _makereal(x :: Vector, tol=1e-10)
    #The eigenvalues may have small imaginary parts
    imt = sum(imag.(x))
    1e-14 < imt <= tol ? @warn("sum of imaginary parts of vector between 1e-14 and tolerance") : nothing
    imt >= tol ? throw(ErrorException("Total imaginary part of x is over tolerance")) : nothing
    return real.(x)
end

"""
    This function returns the Kraus operators of a channel being ran twice in parallel.
"""
function _parallelchan(Ak,Bk)
    Ak2 = Matrix{Any}[]
    Bk2 = Matrix{Any}[]
    for i in eachindex(Ak)
        for j in eachindex(Ak)
            push!(Ak2, kron(Ak[i], Ak[j]))
            push!(Bk2, kron(Bk[i], Bk[j]))
        end
    end
    return Ak2, Bk2
end

"""
    This function returns the Kraus operators of the qubit depolarizing channel with parameter q
"""
function _depolkraus(q)
    idMat = [1 0; 0 1]
        sigmaX = [0 1; 1 0]
        sigmaY = [0 -1im; 1im 0]
        sigmaZ = [1 0; 0 -1]
        Ak = [sqrt(1 - 3 * q / 4) * idMat, sqrt(q / 4) * sigmaX, sqrt(q / 4) * sigmaY, sqrt(q / 4) * sigmaZ]
        Bk = Ak
        return Ak, Bk
end