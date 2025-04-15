using Test, LinearAlgebra
using OperatorMonotoneCorrelationTools

@testset ".src/functions.jl" begin

    @testset "partialtrace" begin
        dA = 2
        dB = 2
        T = reshape(collect(1:1:(dA*dB)^2), (dA * dB, dA * dB))'
        TA = partialtrace(T, dA, dB, 2)
        TB = partialtrace(T, dA, dB, 1)
        TAtrue = [7 11; 23 27]
        TBtrue = [12 14; 20 22]
        @test isapprox(TA,TAtrue,atol=1e-12)
        @test isapprox(TB,TBtrue,atol=1e-12)

        dA = 2
        dB = 3
        T = reshape(collect(1:1:(dA*dB)^2), (dA * dB, dA * dB))'
        TA = partialtrace(T, dA, dB, 2)
        TB = partialtrace(T, dA, dB, 1)
        TAtrue = [24 33; 78 87]
        TBtrue = [23 25 27; 35 37 39; 47 49 51]
        @test isapprox(TA,TAtrue,atol=1e-12)
        @test isapprox(TB,TBtrue,atol=1e-12)

        dA = 3
        dB = 2
        T = reshape(collect(1:1:(dA*dB)^2), (dA * dB, dA * dB))'
        TA = partialtrace(T, dA, dB, 2)
        TB = partialtrace(T, dA, dB, 1)
        TAtrue = [9 13 17; 33 37 41; 57 61 65]
        TBtrue = [45 48; 63 66]
        @test isapprox(TA,TAtrue,atol=1e-12)
        @test isapprox(TB,TBtrue,atol=1e-12)
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

    @testset "returntocompunitary" begin
        recoverscompbasis = true
        for d in [2, 5, 10, 20]
            for run = 1:10
                A = hsrandomstate(d) #initially expressed in comp basis
                #Rewrites A in its eigenbasis
                λ, basis = eigen(A)
                Ap = zeros(d, d)
                for i in 1:d
                    Ap[i, i] = λ[i]
                end
                U = returntocompunitary(A)

                Aback = U * Ap * U'

                isapprox(A, Aback, atol=1e-10) ? nothing : recoverscompbasis = false
            end
        end
        @test recoverscompbasis

        recoverscompbasis = true
        for d in [2, 5, 10, 20]
            for run = 1:10
                A = hsrandomstate(d) #initially expressed in comp basis
                C = hsrandomstate(d) #expressed in comp basis
                Cp = basischange(C, A) #expresses C in the basis of A
                U = returntocompunitary(A)
                Cback = U * Cp * U'
                isapprox(C, Cback, atol=1e-10) ? nothing : recoverscompbasis = false
            end
        end
        @test recoverscompbasis
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

    @testset "choitokraus" begin
        #The Choi operator is numerically controlled by the eigen function,
        #so instead we check the action of using the Choi operators

        #Testing the identity channel
        worksforid = true
        for d = 2:4
            dA = d
            dB = d
            idMat = Matrix(1I, d, d)
            idChoi = vec(idMat) * vec(idMat)'
            Ak, Bk = choitokraus(idChoi, d, d)
            X = reshape(collect(1:1:d^2), (d, d))
            isapprox(X, krausaction(Ak,Bk,X), atol=1e-6) ? nothing : worksforid
        end
        @test worksforid

        #Testing the depolarizing channel
        worksfordepol = true
        for p = 0:0.1:1
            for d = 2:4
                dA = d
                dB = d
                idMat = Matrix(1I, d, d)
                idChoi = vec(idMat) * vec(idMat)'
                maxDepChoi = 1 / d * Matrix(1I, d^2, d^2)
                pDepChoi = (1 - p) * idChoi + p * maxDepChoi
                Ak, Bk = choitokraus(pDepChoi, d, d)

                for i = 1:20
                    ρin = hsrandomstate(d)
                    
                    rhoouttrue = (1 - p) * ρin + p * 1 / d * idMat
                    isapprox(krausaction(Ak,Bk,ρin), rhoouttrue, atol=1e-6) ? nothing : worksfordepol = false
                end
            end
        end
        @test worksfordepol

        #Testing the transpose channel, since we need this to work for arbitrary linear maps
        worksforlinmap = true
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
                rhoouttrue = transpose(rhoin)
                isapprox(krausaction(Ak,Bk,rhoin), rhoouttrue, atol=1e-6) ? nothing : worksforlinmap = false
            end
        end
        @test worksforlinmap
    end

    @testset "isPSD" begin
        X = [1 0 ; 0 1]
        @test isPSD(X)

        X = [1 0 ; 0 -1]
        @test ~isPSD(X)
    end
end