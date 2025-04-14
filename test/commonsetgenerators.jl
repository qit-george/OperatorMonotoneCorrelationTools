using Test, LinearAlgebra
using OperatorMonotoneCorrelationTools

@testset ".src/randomobjects.jl" begin

    @testset "gencompbasis" begin
        d = 1
        basis = gencompbasis(d)
        @test isapprox(basis[1], [1], atol=1e-6)

        d = 2
        basis = gencompbasis(d)
        @test isapprox(basis[1], [1; 0], atol=1e-6)
        @test isapprox(basis[2], [0; 1], atol=1e-6)

        d = 3
        basis = gencompbasis(d)
        @test isapprox(basis[1], [1; 0; 0], atol=1e-6)
        @test isapprox(basis[2], [0; 1; 0]; 0, atol=1e-6)
        @test isapprox(basis[3], [0; 0; 1]; 0, atol=1e-6)
    end

    @testset "genGellMann" begin
        #First we check it returns the correct set on a qubit system
        d = 2
        mats = genGellMann(d)
        @test isapprox(mats[1], 1 / sqrt(2) * [1 0; 0 -1], atol=1e-7)
        @test isapprox(mats[2], 1 / sqrt(2) * [0 1; 1 0], atol=1e-7)
        @test isapprox(mats[3], 1 / sqrt(2) * [0 -1im; 1im 0], atol=1e-7)

        #Then we just check it is an ONB of traceless Hermitian operators for 
        # a few larger dimensions
        for d = 3:6
            istraceless = true
            isnormal = true
            isorthogonal = true
            mats = genGellMann(d)
            for i = 1:d^2-1
                for j = 1:d^2-1
                    if i == j
                        isapprox(tr(mats[i]), 0, atol=1e-7) ? nothing : istraceless = false
                        isapprox(tr(mats[i]' * mats[i]), 1, atol=1e-7) ? nothing : isnormal = false
                    else
                        isapprox(tr(mats[i]' * mats[j]), 0, atol=1e-7) ? nothing : isorthogonal = false
                    end
                end
            end
            @test istraceless && isnormal && isorthogonal
        end
    end

    @testset "genNormDiscWeyl" begin
        #First we check it returns the correct set on a qubit system
        d = 2
        mats = genNormDiscWeyl(d)
        @test isapprox(mats[1], 1 / sqrt(2) * [1 0; 0 1], atol=1e-7)
        @test isapprox(mats[2], 1 / sqrt(2) * [1 0; 0 -1], atol=1e-7)
        @test isapprox(mats[3], 1 / sqrt(2) * [0 1; 1 0], atol=1e-7)
        @test isapprox(mats[4], 1 / sqrt(2) * [0 -1; 1 0], atol=1e-7)

        #Then we just check it is an ONB for a few larger dimensions
        for d = 3:6
            istraceless = true
            isnormal = true
            isorthogonal = true
            mats = genGellMann(d)
            for i = 1:d^2-1
                for j = 1:d^2-1
                    if i == j
                        isapprox(tr(mats[i]' * mats[i]), 1, atol=1e-7) ? nothing : isnormal = false
                    else
                        isapprox(tr(mats[i]' * mats[j]), 0, atol=1e-7) ? nothing : isorthogonal = false
                    end
                end
            end
            @test isnormal && isorthogonal
        end
    end
end