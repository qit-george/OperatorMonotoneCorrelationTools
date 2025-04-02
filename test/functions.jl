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
        idMat = Matrix(1I,d,d)
        idChoi = vec(idMat)*vec(idMat)'
        Ak,Bk = choitokraus(idChoi,d,d)
        X = reshape(collect(1:1:d^2),(d,d))
        @test isapprox(X,Ak[1]*X*Bk[1]', atol=1e-6)
    end

    #Testing the depolarizing channel
    for p = 0:0.1:1
        for d = 2:4
            dA = d;
            dB = d;
            idMat = Matrix(1I,d,d);
            idChoi = vec(idMat)*vec(idMat)';
            maxDepChoi = 1/d*Matrix(1I,d^2,d^2);
            pDepChoi = (1-p)*idChoi + p*maxDepChoi;
            Ak,Bk = choitokraus(pDepChoi,d,d);
    
            for i = 1:d
                rhoin = zeros(d,d);
                rhoin[1,1] = 1;
                rhoout = zeros(d,d);
    
                #This is to avoid relying on kraus action function
                for i in 1:length(Ak)
                    rhoout = rhoout + Ak[i]*rhoin*Bk[i]';
                end
                rhoouttrue = (1-p)*rhoin + p*1/d*idMat ;
                @test isapprox(rhoout,rhoouttrue,atol=1e-6)
            end
        end
    end

    #Testing the transpose channel, since we need this to work for arbitrary linear maps
    for d = 2:4
        dA = d
        dB = d
        #Build the Choi operator
        transposeChoi = zeros(d^2,d^2)
        for i = 1:d
            for j = 1:d
                Eij = zeros(d,d)
                Eij[i,j] = 1
                transposeChoi = transposeChoi + kron(Eij,transpose(Eij))
            end
        end
        #Get Kraus operators 
        Ak, Bk = choitokraus(transposeChoi,d,d)

        #random tests
        for t = 1:10
            rhoin = rand(Complex{Float64}, (d,d))
            rhoout = zeros(d,d)
            for i in 1:length(Ak)
                rhoout = rhoout + Ak[i]*rhoin*Bk[i]';
            end

            rhoouttrue = transpose(rhoin)
            @test isapprox(rhoout,rhoouttrue,atol=1e-6)
        end
    end
end




end

end