"""
    BinnedData

A data structure containing a time-binned dataframe. Fields:

- `data`: the dataframe
- `bin_resolution`: binning resolution (see `bin_time`)
- `midpoints`: midpoints of the time bins on the original time scale
"""
mutable struct BinnedData
    data::DataFrame
    bin_resolution::Int
    midpoints::Vector{Float64}
end

"""
    BinnedTimeSeries

FIXME
"""
mutable struct BinnedTimeSeries
    data::DataFrame
    bin_resolution::Int
    midpoints::Vector{Float64}
end

mutable struct TEPermutationTestResult
    empirical::Float64
    threshold::Float64
    pvalue::Float64
    alpha::Float64
    significant::Bool
    distribution::Union{Nothing, Vector{Float64}}
end
