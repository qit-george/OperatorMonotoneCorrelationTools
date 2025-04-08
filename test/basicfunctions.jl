using Test, LinearAlgebra
using OperatorMonotoneCorrelationTools

@testset ".src/functions.jl" begin

    @testset "choitokraus" begin
        #There is some freedom in the Choi operators,
        #so instead we check the action of using the Choi operators

        #Testing the identity channel
        for d = 2:4
            dA = d
            dB = d
            idMat = Matrix(1I, d, d)
            idChoi = vec(idMat) * vec(idMat)'
            Ak, Bk = choitokraus(idChoi, d, d)
            X = reshape(collect(1:1:d^2), (d, d))
            @test isapprox(X, Ak[1] * X * Bk[1]', atol=1e-6)
        end

        #Testing the depolarizing channel
        for p = 0:0.1:1
            for d = 2:4
                dA = d
                dB = d
                idMat = Matrix(1I, d, d)
                idChoi = vec(idMat) * vec(idMat)'
                maxDepChoi = 1 / d * Matrix(1I, d^2, d^2)
                pDepChoi = (1 - p) * idChoi + p * maxDepChoi
                Ak, Bk = choitokraus(pDepChoi, d, d)

                for i = 1:d
                    rhoin = zeros(d, d)
                    rhoin[1, 1] = 1
                    rhoout = zeros(d, d)

                    #This is to avoid relying on kraus action function
                    for i in 1:length(Ak)
                        rhoout = rhoout + Ak[i] * rhoin * Bk[i]'
                    end
                    rhoouttrue = (1 - p) * rhoin + p * 1 / d * idMat
                    @test isapprox(rhoout, rhoouttrue, atol=1e-6)
                end
            end
        end

        #Testing the transpose channel, since we need this to work for arbitrary linear maps
        for d = 2:4
            dA = d
            dB = d
            #Build the Choi operator
            transposeChoi = zeros(d^2, d^2)
            for i = 1:d
                for j = 1:d
                    Eij = zeros(d, d)
                    Eij[i, j] = 1
                    transposeChoi = transposeChoi + kron(Eij, transpose(Eij))
                end
            end
            #Get Kraus operators 
            Ak, Bk = choitokraus(transposeChoi, d, d)

            #random tests
            for t = 1:10
                rhoin = rand(Complex{Float64}, (d, d))
                rhoout = zeros(d, d)
                for i in 1:length(Ak)
                    rhoout = rhoout + Ak[i] * rhoin * Bk[i]'
                end

                rhoouttrue = transpose(rhoin)
                @test isapprox(rhoout, rhoouttrue, atol=1e-6)
            end
        end
    end

    @testset "krausaction" begin
        #Testing the identity channel
        for d = 2:5
            idMat = Matrix(I, d, d)
            Ak = Matrix{Any}[]
            Bk = Matrix{Any}[]
            push!(Ak, idMat)
            push!(Bk, idMat)
            X = reshape(collect(1:1:d^2), (d, d))

            @test isapprox(krausaction(Ak, Bk, X), X, atol=1e-6)
        end

        #Testing the qubit depolarizing channel
        for p = 0:0.1:1
            idMat = Matrix(I, 2, 2)
            sigmaX = [0 1; 1 0]
            sigmaY = [0 -1im; 1im 0]
            sigmaZ = [1 0; 0 -1]
            Ak = [sqrt(1 - 3 * p / 4) * idMat, sqrt(p / 4) * sigmaX, sqrt(p / 4) * sigmaY, sqrt(p / 4) * sigmaZ]
            Bk = Ak
            rhoin = [1 0; 0 0]
            rhoouttrue = (1 - p) * rhoin + p / 2 * idMat
            rhoout = krausaction(Ak, Bk, rhoin)
            @test isapprox(rhoout, rhoouttrue, atol=1e-6)
        end

        #Testing the transpose channel to check it works with linear maps more generally
        for d = 2:5
            Ak = Matrix{Any}[]
            Bk = Matrix{Any}[]
            for i = 1:d
                for j = 1:d
                    Eij = zeros(d, d)
                    Eij[i, j] = 1
                    push!(Ak, Eij)
                    push!(Bk, transpose(Eij))
                end
            end
            X = reshape(collect(1:1:d^2), (d, d))
            @test isapprox(transpose(X), krausaction(Ak, Bk, X), atol=1e-6)
        end
    end

    @testset "basischange" begin
        #check it throws errors properly
        A = [1 2 5; 3 4 6]
        B = [1 2im; -2im 3]
        @test_throws ArgumentError basischange(A, B)
        A = [1 2; 3 4]
        B = [1 2im; 2im 3]
        @test_throws ArgumentError basischange(A, B)

        #check it changes bases properly
        #checking for Pauli bases
        A = [1 0; 0 0]
        B = [0 1; 1 0] #sigmaX
        @test isapprox(basischange(A, B), 1 / 2 * [1 -1; -1 1], atol=1e-6)
        A = [0 0; 0 1]
        @test isapprox(basischange(A, B), 1 / 2 * [1 1; 1 1], atol=1e-6)
        B = [0 -1im; 1im 0] #sigmaY
        @test isapprox(basischange(A, B), 1 / 2 * [1 -1; -1 1], atol=1e-6)
        A = [1 0; 0 0]
        @test isapprox(basischange(A, B), 1 / 2 * [1 1; 1 1], atol=1e-6)

        A = [1 2 3; 4 5 6; 7 8 9]
        B = [0 1 0; 1 0 0; 0 0 0]
        @test isapprox(basischange(A, B), [0 -3/sqrt(2) -3; -1/sqrt(2) 9 15/sqrt(2); -1 9/sqrt(2) 6], atol=1e-6)
    end

    @testset "Haarrandomunitary" begin
        #There is not much we can check, so we just
        #test they are actually unitaries
        for i in 1:1:100
            d = rand(1:100)
            U = Haarrandomunitary(d)
            @test isapprox(U'*U,I(d),atol=1e-6)
            @test isapprox(U*U',I(d),atol=1e-6)
        end
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

end