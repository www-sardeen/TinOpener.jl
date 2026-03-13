"""
    BinnedTimeSeries

A data structure containing a time series produced by binning
a dataframe. Fields:

- `data`: the time series
- `bin_resolution`: binning resolution (see [`bin_time`](@ref))
- `midpoints`: midpoints of the time bins on the original time scale
"""
mutable struct BinnedTimeSeries
    data::DataFrame
    bin_resolution::Int
    midpoints::Vector{Float64}
end


"""
    binary_to_proportion(x::BinnedData, variable, ref_level;
                         shuffle = false, α = 0)

Transforms a binary categorical variable into its proportion, 
calculated per time bin.

The `variable` is first transformed into a vector of ones and zeroes;
whenever the value coincides with `ref_level`, we replace the value with
a one, otherwise we replace it with a zero. The proportion of `ref_level`
per time bin is then calculated as the mean over this vector of ones and
zeroes.

If `shuffle` is `true`, a bootstrap (random sample with replacement) is
taken per each time bin.

The keyword argument `α` can be used to perform additive smoothing. For
instance, setting `α = 1` gives add-one (Laplace) smoothing.

## Value

The output is a [`BinnedTimeSeries`](@ref).
"""
function binary_to_proportion(x::BinnedData, variable, ref_level;
        shuffle = false, α = 0)
    data = copy(x.data)

    # make response variable numeric
    data.resp = ifelse.(data[!, variable] .== ref_level, 1, 0)

    # obtain means per dates
    if shuffle
        data = @pipe groupby(data, :timebin) |> combine(_, bootstrap)
    end

    data = @pipe groupby(data, :timebin) |> combine(_,
                                                    :resp => (r -> sum(r .== 0) + α) => :value_0,
                                                    :resp => (r -> sum(r .== 1) + α) => :value_1)

    data.n = data.value_0 .+ data.value_1
    data.value = data.value_1 ./ data.n

    return BinnedTimeSeries(data, x.bin_resolution, x.midpoints)
end


"""
    expand_missing(x::BinnedTimeSeries)

Expands the data in a [`BinnedTimeSeries`](@ref) so that time bins with no data
are explicitly flagged as having `missing` proportions.
"""
function expand_missing(x::BinnedTimeSeries)
    @pipe leftjoin(DataFrame(timebin = 1:length(x.midpoints)), x.data; on = :timebin) |> sort(_, :timebin) |> BinnedTimeSeries(_, x.bin_resolution, x.midpoints)
end
