module TinOpener

# usings and imports
using CategoricalArrays
using DataFrames
using Pipe

import Dates
import StatsBase
import TSML

# source files
include("structs.jl")
include("binning.jl")
include("bootstrap.jl")
include("imputation.jl")
include("proportions.jl")

# exports (structs)
export BinnedCorpus
export BinnedTimeSeries

# exports (functions)
export bin_time
export bootstrap
export categorical_to_proportion
export impute

end
