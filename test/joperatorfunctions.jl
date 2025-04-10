using Test, LinearAlgebra
using OperatorMonotoneCorrelationTools

@testset ".src/joperatorfunctions.jl" begin

    @testset "perspective" begin
        #Testing f(x)=x
        function f(x)
            return x
        end
        x, y = 1, -2
        @test_throws ArgumentError perspective(x, y, f, 0, 1)
        x, y = -1, 2
        @test_throws ArgumentError perspective(x, y, f, 0, 1)
        x, y = 1, 2
        @test perspective(x, y, f, 0, 1) == 1
        x, y = 1, 0
        @test perspective(x, y, f, 0, 1) == 1
        x, y = 0, 1
        @test perspective(x, y, f, 0, 1) == 0

        #Testing f(x) = sqrt(x)
        function f(x)
            return sqrt(x)
        end
        x, y = 1, -2
        @test_throws ArgumentError perspective(x, y, f, 0, 1)
        x, y = -1, 2
        @test_throws ArgumentError perspective(x, y, f, 0, 1)
        x, y = 1, 2
        @test isapprox(perspective(x, y, f, 0, 0), sqrt(2), atol=1e-6)
        x, y = 3, 0
        @test perspective(x, y, f, 0, 0) == 0
        x, y = 0, 4
        @test perspective(x, y, f, 0, 0) == 0
    end

    @testset "innerproductf" begin
        #If sigma is uniform and f(1)=1, then it is the HS inner product up to (1/d)^p
        function f(x)
            return x
        end
        for d = 1:4
            for p = -2:1:2
                for t = 1:5
                    sigma = 1 / d * Matrix(I, d, d)
                    X = rand(d, d)
                    Y = rand(d, d)
                    t = innerproductf(X, Y, sigma, p, f, 0, 1)
                    ttrue = (1 / d)^(p) * tr(X' * Y)
                    @test isapprox(t, ttrue, atol=1e-10)
                end
            end
        end
        function f(x)
            return sqrt(x)
        end
        for d = 1:4
            for p = -2:1:2
                for t = 1:5
                    sigma = 1 / d * Matrix(I, d, d)
                    X = rand(d, d)
                    Y = rand(d, d)
                    t = innerproductf(X, Y, sigma, p, f, 0, 0)
                    ttrue = (1 / d)^(p) * tr(X' * Y)
                    @test isapprox(t, ttrue, atol=1e-10)
                end
            end
        end
        #tests arithmetic mean
        function f(x)
            return (x + 1) / 2
        end
        for d = 1:4
            for p = -2:1:2
                for t = 1:5
                    sigma = 1 / d * Matrix(I, d, d)
                    X = rand(d, d)
                    Y = rand(d, d)
                    t = innerproductf(X, Y, sigma, p, f, 1 / 2, 1 / 2)
                    ttrue = (1 / d)^(p) * tr(X' * Y)
                    @test isapprox(t, ttrue, atol=1e-10)
                end
            end
        end
        #one case where σ is non-uniform
        σ = [1/2 -1/4; -1/4 1/2]
        X = [1 2; 3 5]
        Y = [5 6; 7 4]
        for p = -2:1/2:2
            function f(x)
                return x
            end
            ttrue = 11^(2) / 2 * (1 / 4)^(p) - 3 / 2 * (3 / 4)^(p) * f(1 / 3)^(p) - (3 / 4)^(p)
            @test isapprox(innerproductf(X, Y, σ, p, f, 0, 1), ttrue, atol=1e-6)
            function f(x)
                return sqrt(x)
            end
            ttrue = 11^(2) / 2 * (1 / 4)^(p) - 3 / 2 * (3 / 4)^(p) * f(1 / 3)^(p) - (3 / 4)^(p)
            @test isapprox(innerproductf(X, Y, σ, p, f, 0, 0), ttrue, atol=1e-6)
            function f(x)
                return (x + 1) / 2
            end
            ttrue = 11^(2) / 2 * (1 / 4)^(p) - 3 / 2 * (3 / 4)^(p) * f(1 / 3)^(p) - (3 / 4)^(p)
            @test isapprox(innerproductf(X, Y, σ, p, f, 1 / 2, 1 / 2), ttrue, atol=1e-6)
        end
    end

    @testset "Jfpsigma" begin
        #If sigma is uniform, the operator is just rescaled by (1/d)^p
        function f(x)
            return x
        end
        f0 = 0
        fpinf = 1
        for d = 2:5
            for p = -2:1/2:2
                sigma = 1 / d * Matrix(I, d, d)
                Y = rand(d, d)

                Yout = Jfpsigma(Y, sigma, p, f, f0, fpinf)
                Ytrue = (1 / d)^(p) * Y
                @test isapprox(Yout, Ytrue, atol=1e-10)
            end
        end

        #A non-uniform example
        for i = 1:2
            if i == 1
                function f(x)
                    return sqrt(x)
                end
                f0 = 0
                fpinf = 0
            else
                function f(x)
                    return (x + 1) / 2
                end
                f0 = 1 / 2
                fpinf = 1 / 2
            end
            for p = -2:1/2:2
                σ = [1/2 -1/4; -1/4 1/2]
                Y = [5 6; 7 4]
                #Ytrue = [11*perspective(1 / 4, 1 / 4, f, f0, fpinf)^p perspective(1 / 4, 3 / 4, f, f0, fpinf)^p; 0 -2*perspective(3 / 4, 3 / 4, f, f0, fpinf)^p]
                a1 = perspective(1 / 4, 1 / 4, f, f0, fpinf)^p
                a2 = perspective(1 / 4, 3 / 4, f, f0, fpinf)^p
                a3 = perspective(3 / 4, 3 / 4, f, f0, fpinf)^p

                e11 = 11 / 2 * a1 + a2 / 2 - a3
                e12 = 11 / 2 * a1 - a2 / 2 + a3
                e21 = 11 / 2 * a1 + a2 / 2 + a3
                e22 = 11 / 2 * a1 - a2 / 2 - a3

                Ytrue = [e11 e12; e21 e22]
                Yout = Jfpsigma(Y, σ, p, f, f0, fpinf)
                @test isapprox(Yout, Ytrue, atol=1e-10)
            end
        end
    end

    @testset "getONB" begin
        #In the case σ=π (max mixed), an ONB just gets rescaled by d^(p/2)
        #So we check this for two functions first
        function f(x)
            return x
        end
        f0 = 0
        fpinf = 1
        for d = 2:5
            for p = -2:0.5:2
                σ = 1 / d * Matrix(I, d, d)
                onb = getONB(σ, p, f, f0, fpinf)

                isnormal = true
                isorthogonal = true
                for i in eachindex(onb)
                    for j in eachindex(onb)
                        t = innerproductf(onb[i], onb[j], σ, p, f, f0, fpinf)
                        if i == j
                            isapprox(t, 1, atol=1e-6) ? nothing : isnormal = false
                        else
                            isapprox(t, 0, atol=1e-6) ? nothing : isorthogonal = false
                        end
                    end
                end

                d = size(σ)[1]
                onbold = genGellMann(d)
                pushfirst!(onbold, sqrt(σ))
                #Check is what it should be analytically
                iscorrect = true
                for i in eachindex(onb)
                    isapprox(onb[i], d^(p / 2) * onbold[i], atol=1e-6) ? nothing : iscorrect = false
                end

                @test isnormal && isorthogonal && iscorrect
            end
        end

        #same thing but with different function
        function f(x)
            return (x + 1) / 2
        end
        f0 = 1 / 2
        fpinf = 1 / 2
        for d = 2:5
            for p = -2:0.5:2
                σ = 1 / d * Matrix(I, d, d)
                onb = getONB(σ, p, f, f0, fpinf)

                isnormal = true
                isorthogonal = true
                for i in eachindex(onb)
                    for j in eachindex(onb)
                        t = innerproductf(onb[i], onb[j], σ, p, f, f0, fpinf)
                        if i == j
                            isapprox(t, 1, atol=1e-6) ? nothing : isnormal = false
                        else
                            isapprox(t, 0, atol=1e-6) ? nothing : isorthogonal = false
                        end
                    end
                end

                d = size(σ)[1]
                onbold = genGellMann(d)
                pushfirst!(onbold, sqrt(σ))
                #Check is what it should be analytically
                iscorrect = true
                for i in eachindex(onb)
                    isapprox(onb[i], d^(p / 2) * onbold[i], atol=1e-6) ? nothing : iscorrect = false
                end

                @test isnormal && isorthogonal && iscorrect
            end
        end

        #Now we do the same thing, but with random states, so we
        #just check that it generates ONBs
        function f(x)
            return x
        end
        f0 = 0
        fpinf = 1
        for d = 2:5
            for p = -2:1:2
                σ = hsrandomstate(d)
            
                onb = getONB(σ, p, f, f0, fpinf)
        
                isnormal = true
                isorthogonal = true
                for i in eachindex(onb)
                    for j in eachindex(onb)
                        t = innerproductf(onb[i], onb[j], σ, p, f, f0, fpinf)
                        if i == j
                            isapprox(t, 1, atol=1e-6) ? nothing : isnormal = false
                        else
                            isapprox(t, 0, atol=1e-6) ? nothing : isorthogonal = false
                        end
                    end
                end
        
                @test isnormal && isorthogonal
            end
        end
        
        #Same thing but with different function
        function f(x)
            return (x+1)/2
        end
        f0 = 1/2
        fpinf = 1/2
        for d = 2:5
            for p = -2:1:2
                σ = hsrandomstate(d)
            
                onb = getONB(σ, p, f, f0, fpinf)
        
                isnormal = true
                isorthogonal = true
                for i in eachindex(onb)
                    for j in eachindex(onb)
                        t = innerproductf(onb[i], onb[j], σ, p, f, f0, fpinf)
                        if i == j
                            isapprox(t, 1, atol=1e-6) ? nothing : isnormal = false
                        else
                            isapprox(t, 0, atol=1e-6) ? nothing : isorthogonal = false
                        end
                    end
                end
        
                @test isnormal && isorthogonal
            end
        end
    end

    @testset "SchReversalMap" begin
        #As we know the Schrodinger reversal map 𝒮_{f,ℰ,σ}(ℰ(σ)) = σ,
        #we verify the function does that on random quantum states
        
        function f(x)
            return x
        end
        f0 = 0
        fpinf = 1
        returnsstate = true
        for q = 0:0.01:1
            idMat = [1 0; 0 1]
            sigmaX = [0 1; 1 0]
            sigmaY = [0 -1im; 1im 0]
            sigmaZ = [1 0; 0 -1]
            Ak = [sqrt(1 - 3 * q / 4) * idMat, sqrt(q / 4) * sigmaX, sqrt(q / 4) * sigmaY, sqrt(q / 4) * sigmaZ]
            Bk = Ak
            for run = 1:10
                σ = hsrandomstate(2)
                σout = krausaction(Ak, Bk, σ)

                step3 = SchReversalMap(σout, Ak, Bk, σ, f, f0, fpinf)
                norm(step3 - σ) < 1e-14 ? nothing : returnsstate = false
            end
        end
        @test returnsstate
        
        function f(x)
            return sqrt(x)
        end
        f0 = 0
        fpinf = 0
        returnsstate = true
        for q = 0:0.01:1
            idMat = [1 0; 0 1]
            sigmaX = [0 1; 1 0]
            sigmaY = [0 -1im; 1im 0]
            sigmaZ = [1 0; 0 -1]
            Ak = [sqrt(1 - 3 * q / 4) * idMat, sqrt(q / 4) * sigmaX, sqrt(q / 4) * sigmaY, sqrt(q / 4) * sigmaZ]
            Bk = Ak
            for run = 1:10
                σ = hsrandomstate(2)
                σout = krausaction(Ak, Bk, σ)

                step3 = SchReversalMap(σout, Ak, Bk, σ, f, f0, fpinf)
                norm(step3 - σ) < 1e-14 ? nothing : returnsstate = false
            end
        end
        @test returnsstate

        function f(x)
            return (x+1)/2
        end
        f0 = 1/2
        fpinf = 1/2
        returnsstate = true
        for q = 0:0.01:1
            idMat = [1 0; 0 1]
            sigmaX = [0 1; 1 0]
            sigmaY = [0 -1im; 1im 0]
            sigmaZ = [1 0; 0 -1]
            Ak = [sqrt(1 - 3 * q / 4) * idMat, sqrt(q / 4) * sigmaX, sqrt(q / 4) * sigmaY, sqrt(q / 4) * sigmaZ]
            Bk = Ak
            for run = 1:10
                σ = hsrandomstate(2)
                σout = krausaction(Ak, Bk, σ)

                step3 = SchReversalMap(σout, Ak, Bk, σ, f, f0, fpinf)
                norm(step3 - σ) < 1e-14 ? nothing : returnsstate = false
            end
        end
        @test returnsstate
    end

    @testset "getcontractioncoeff" begin
        #In "Quantum Rényi and f-divergences from integral representations" by 
        #Hirche and Tomamichel it is shown that for a generalized depolarizing channel
        #the input-dependent contraction coefficient for f = f_{LM} is (1-q)^2. 
        #Here we verify our function achieves this

        function f(x)
            if x != 1 && x > 0
                return (x - 1) / log(x)
            elseif x == 1
                return 1
            else
                throw(ArgumentError("x cannot be negative"))
            end
        end
        f0 = 0
        fpinf = 0

        #Testing qubit depolarizing channel
        maxdif = 0
        σ = [1/2 0; 0 1/2]
        worksfordepol = true
        for q = 0:0.05:1
            idMat = [1 0; 0 1]
            sigmaX = [0 1; 1 0]
            sigmaY = [0 -1im; 1im 0]
            sigmaZ = [1 0; 0 -1]
            Ak = [sqrt(1 - 3 * q / 4) * idMat, sqrt(q / 4) * sigmaX, sqrt(q / 4) * sigmaY, sqrt(q / 4) * sigmaZ]
            Bk = Ak
            calc = getcontractioncoeff(Ak, Bk, σ, f, f0, fpinf)
            calctrue = (1 - q)^2
            isapprox(calc, calctrue, atol=1e-12) ? nothing : worksfordepol = false
        end
        @test worksfordepol

        #testing qutrit generalized depolarizing channel with random state
        d = 3
        worksfordepol = true
        for q = 0:0.05:1
            for run = 1:4
                σ = hsrandomstate(d)
                Φv = vec(Matrix(1I, d, d))
                Φ = Φv * Φv'
                choimat = (1 - q)Φ + q * kron(Matrix(1I, d, d), σ)
                Ak, Bk = choitokraus(choimat, d, d)
                calc = getcontractioncoeff(Ak, Bk, σ, f, f0, fpinf)
                calctrue = (1 - q)^2
                val = abs(calc - calctrue)
                isapprox(calc, calctrue, atol=1e-10) ? nothing : worksfordepol = false
            end
        end
        @test worksfordepol
    end
end