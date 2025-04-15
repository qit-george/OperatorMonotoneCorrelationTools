using Test, SafeTestsets

println("Importing OperatorMonotoneCorrelationTools")

@time using OperatorMonotoneCorrelationTools

println("running ./test/OperatorMonotoneCorrelationTools.jl")

@time begin 

    @testset "running ./test/runtests.jl" begin

        println("testing ./src/basicfunctions.jl")
        @time @safetestset ".test/basicfunctions.jl" begin include("basicfunctions.jl") end

        println("testing ./src/randomobjects.jl")
        @time @safetestset ".test/randomobjects.jl" begin include("randomobjects.jl") end

        println("testing ./src/commonsetgenerators.jl")
        @time @safetestset ".test/commonsetgenerators.jl" begin include("commonsetgenerators.jl") end

        println("testing ./src/joperatorfunctions.jl")
        @time @safetestset ".test/joperatorfunctions.jl" begin include("joperatorfunctions.jl") end

    end
end
