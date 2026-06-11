module TinOpener

# usings and imports
using Associations
using CategoricalArrays
using DataFrames
using Distributed
using Pipe
using Optimization
import Crayons
import ProgressMeter
import OptimizationOptimJL
import ADTypes
import FiniteDiff
import SciMLBase
import SpecialFunctions
import Integrals

import Dates
import Distributions
import LinearAlgebra
#import RCall
import Statistics
import TimeseriesSurrogates
import DataStructures

# source files
#include("structs.jl")

include("binning.jl")
include("bootstrap.jl")
#include("entropies.jl")
#include("imputation.jl")
include("ssm.jl")
include("kalman.jl")
include("kalman_ininvariant.jl")
include("fitting.jl")
include("parallel.jl")
include("proportions.jl")
include("surrogates.jl")
include("utils.jl")
include("distributions.jl")
include("convolutions.jl")

# exports (structs)
export AbstractInvariantLinearGaussianSSM
export AbstractKalmanRecursion
export AbstractBinnedData
export BinnedData
export SqueezedData
export BinnedTimeSeries
export InvariantKalmanRecursion
export MomentumCoupledSLSSM
export SLSSM
export SSMFit
#export TEPermutationTestResult
export KalmanRecursion
export AbstractLinearGaussianSSM
export SSM
export AbstractSSM
export BiBeta
export AbstractModelFit
export ModelParameters

# exports (functions)
export aic
export aicc
#export appell_F1
export bin_time
export bin_time_natural
export bootstrap
export binary_to_proportion
export bzc_kernel
export conditional_entropy
export convolve
export expand_missing
export filter!
export fit_SSM
export fit_restricted_SLSSM
export forward!
export get_filtered_states
export get_filtered_variances
export get_parameters
export get_smoothed_states
export get_smoothed_variances
export kalman_recursion
export logistic
export logit
export loglikelihood
export logpdf
export lrtest
export impute_kalman
export impute_last!
export mapply
export momentmatch
export mutual_information
export numeric_to_mean
export n_free_parameters
export pdf
export prepare_model
export shuffle
export simulate
export smooth!
export strong_kernel
export squeeze_time
export transfer_entropy
export weak_kernel
export zc_kernel
#export TE_permutation_test

end
