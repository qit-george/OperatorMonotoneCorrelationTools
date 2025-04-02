using Test, SafeTestsets

println("Importing OperatorMonotoneCorrelationTools")

@time using OperatorMonotoneCorrelationTools

println("running ./test/OperatorMonotoneCorrelationTools.jl")

@time begin 

    @testset "running ./test/runtests.jl" begin

        println("testing ./src/functions.jl")
        @time @safetestset ".test/functions.jl" begin include("functions.jl") end

    end
end
