module TinOpener

# usings and imports
using Associations
using CategoricalArrays
using DataFrames
using Pipe

import Dates
import Distributions
import LinearAlgebra
import RCall
import Statistics
import TimeseriesSurrogates

# source files
include("structs.jl")

include("binning.jl")
include("bootstrap.jl")
include("entropies.jl")
include("imputation.jl")
include("ssm.jl")
include("kalman.jl")
include("proportions.jl")
include("surrogates.jl")
include("utils.jl")

# exports (structs)
export AbstractInvariantLinearGaussianSSM
export AbstractKalmanRecursion
export BinnedData
export BinnedTimeSeries
export InvariantKalmanRecursion
export MomentumCoupledSLSSM
export SLSSM
export TEPermutationTestResult

# exports (functions)
export aic
export bin_time
export bootstrap
export binary_to_proportion
export expand_missing
export filter!
export forward!
export get_filtered_states
export get_filtered_variances
export get_smoothed_states
export get_smoothed_variances
export kalman_recursion
export logistic
export logit
export loglikelihood
export impute_kalman
export shuffle
export smooth!
export transfer_entropy
export TE_permutation_test

end
