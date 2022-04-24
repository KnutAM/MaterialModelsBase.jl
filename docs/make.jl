using MaterialModelsInterface
using Documenter

DocMeta.setdocmeta!(MaterialModelsInterface, :DocTestSetup, :(using MaterialModelsInterface); recursive=true)

makedocs(;
    modules=[MaterialModelsInterface],
    authors="Knut Andreas Meyer and contributors",
    repo="https://github.com/KnutAM/MaterialModelsInterface.jl/blob/{commit}{path}#{line}",
    sitename="MaterialModelsInterface.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://KnutAM.github.io/MaterialModelsInterface.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/KnutAM/MaterialModelsInterface.jl",
    devbranch="main",
)
