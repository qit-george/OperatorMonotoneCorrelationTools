using Test, LinearAlgebra
using OperatorMonotoneCorrelationTools

@testset ".src/randomobjects.jl" begin

    @testset "Haarrandomunitary" begin
        #There is not much we can check, so we just
        #test they are actually unitaries
        for i in 1:1:100
            d = rand(1:100)
            U = Haarrandomunitary(d)
            @test isapprox(U' * U, I(d), atol=1e-6)
            @test isapprox(U * U', I(d), atol=1e-6)
        end
    end

    @testset "hsrandomstate" begin
        #We take our test from Qetlab
        #namely, purity of ρ drawn from μ_{nk} is
        #given by (n+m)/(nm+1)
        n = 3
        m = 7
        ct = 0
        s = 100000
        for j = 1:s
            ρ = hsrandomstate(n, m)
            ct = ct + real(tr(ρ' * ρ))
        end
        @test isapprox(ct / s, (n + m) / (n * m + 1), atol=1e-3) # average purity
    end

    @testset "randomquantumchannel" begin
        funcworks = true
        for dA = 2:10
            for dB = 2:10
                choi = randomquantumchannel(dA, dB)
                isapprox(partialtrace(choi, dA, dB, 2), Matrix(1I, dA, dA), atol=1e-10) ? nothing : funcworks = false
                isPSD(choi) ? nothing : funcworks
            end
        end
        funcworks
    end
end