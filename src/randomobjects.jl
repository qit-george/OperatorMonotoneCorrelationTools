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
    hsrandomstate(d::Int,k::Int=d, re=false)

Draws a density matrix according to the ``\\mu_{n,k}`` 
distribution. The method of construction follows Lemma 1
of ["Asymptotics of random density matrices" by Ion Nechita](https://arxiv.org/abs/quant-ph/0702154).
In the case the variable re is set to true, the function generates
a random state with only real entries. No analytic structure
about this measure is guaranteed.
"""
function hsrandomstate(d::Int, k::Int=d, re=false)
    #Construct Z that is made up of iid 𝒩_{ℂ}(0,1)
    #which means Z[i,j] = X[i,j] +iY[i,j] where X,Y ∼ 𝒩(0,1/2)
    #thus we need to rescale randn by 1/2
    if re
        Z = sqrt(1 / 2) * randn(d, k)
    else
        Z = sqrt(1 / 2) * randn(d, k) + sqrt(1 / 2) * 1im * randn(d, k)
    end
    return (Z * Z' / tr(Z * Z'))

end

"""
    randomquantumchannel(dA,dB)

Returns the Choi state of a randomquantum channel by drawing ``\\rho_{AB}`` according
to the Hilbert-Schmidt measure and then returning ``\\rho_{A}^{-1/2}\\rho_{AB}\\rho_{A}^{-1/2}``.
If re is set to true, it generates the quantum channel from a quantum state with real values.
"""
function randomquantumchannel(dA,dB,re=false)
    ρAB = hsrandomstate(dA*dB,dA*dB,re)
    ρA = partialtrace(ρAB,dA,dB,2)
    idB = Matrix(1I,dB,dB)
    return kron(ρA^(-1/2),idB)*ρAB*kron(ρA^(-1/2),idB)
end