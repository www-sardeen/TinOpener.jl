module TinOpener

# usings and imports
using Associations
using CategoricalArrays
using DataFrames
using Pipe

import Dates
import Distributions
import RCall
import Statistics
import TimeseriesSurrogates

# source files
include("structs.jl")

include("binning.jl")
include("bootstrap.jl")
include("entropies.jl")
include("imputation.jl")
include("proportions.jl")
include("surrogates.jl")
include("utils.jl")

# exports (structs)
export BinnedData
export BinnedTimeSeries
export TEPermutationTestResult

# exports (functions)
export bin_time
export bootstrap
export binary_to_proportion
export expand_missing
export logistic
export logit
export impute_kalman
export shuffle
export transfer_entropy
export TE_permutation_test

end
