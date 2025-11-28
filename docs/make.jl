using Documenter, TinOpener

makedocs(sitename = "TinOpener.jl",
         pages = [
                  "Home" => "index.md",
                  "ref.md"
                 ],
         remotes = nothing)
