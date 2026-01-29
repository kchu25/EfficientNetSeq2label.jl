using EfficientNetSeq2label
using Documenter

DocMeta.setdocmeta!(EfficientNetSeq2label, :DocTestSetup, :(using EfficientNetSeq2label); recursive=true)

makedocs(;
    modules=[EfficientNetSeq2label],
    authors="Shane Kuei-Hsien Chu (skchu@wustl.edu)",
    sitename="EfficientNetSeq2label.jl",
    format=Documenter.HTML(;
        canonical="https://kchu25.github.io/EfficientNetSeq2label.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/kchu25/EfficientNetSeq2label.jl",
    devbranch="main",
)
