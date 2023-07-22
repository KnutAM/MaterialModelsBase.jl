using MaterialModelsBase
using Documenter

DocMeta.setdocmeta!(MaterialModelsBase, :DocTestSetup, :(using MaterialModelsBase); recursive=true)

makedocs(;
    modules=[MaterialModelsBase],
    authors="Knut Andreas Meyer and contributors",
    repo="https://github.com/KnutAM/MaterialModelsBase.jl/blob/{commit}{path}#{line}",
    sitename="MaterialModelsBase.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://KnutAM.github.io/MaterialModelsBase.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Stress states" => "stressiterations.md",
        "Differentiation" => "differentiation.md"
    ],
)

deploydocs(;
    repo="github.com/KnutAM/MaterialModelsBase.jl",
    devbranch="main",
    push_preview=true,
)
