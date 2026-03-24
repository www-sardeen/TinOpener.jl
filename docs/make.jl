using Documenter, TinOpener

makedocs(sitename = "TinOpener.jl",
         pages = [
                  "Home" => "index.md",
                  "Reference" => [
                                  "General time series tools" => "ts.md",
                                  "Kalman filtering" => "kalman.md",
                                  "Parallel processing" => "parallel.md",
                                  "State space models" => "ssm.md",
                                  "Utilities" => "utils.md",
                                  "Index" => "ref.md"
                                 ]
                 ],
         remotes = nothing)
