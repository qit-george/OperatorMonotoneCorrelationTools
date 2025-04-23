push!(LOAD_PATH,"../src/")
using Documenter, OperatorMonotoneCorrelationTools

makedocs(;
    modules = [OperatorMonotoneCorrelationTools],
    doctest = true,
    linkcheck= true,
    authors = "Ian George <qit.george@gmail.com>",
    sitename = "OperatorMonotoneCorrelationTools",
    pages = [ 
        "Home" => "index.md",
        "Basic Functions" => "basicfunctions.md",
        "JOperator Functions" => "joperatorfunctions.md",
        "Common Set Generators" => "commonsetgenerators.md",
        "Random Objects" => "randomobjects.md",
        "Helper Functions" => "helperfunctions.md",
    ],
)