"""
    BinnedData

A data structure containing a time-binned dataframe. Fields:

- `data`: the dataframe
- `bin_resolution`: binning resolution (see [`bin_time`](@ref))
- `midpoints`: midpoints of the time bins on the original time scale
"""
mutable struct BinnedData
    data::DataFrame
    bin_resolution::Int
    midpoints::Vector{Float64}
end


"""
    bin_time(data::DataFrame; time_variable = "date", bin_resolution = 10)

Bin the time variable in a `DataFrame` into `bin_resolution` equally
spaced bins.

The left endpoint of the first bin is the minimum of `time_variable`;
the right endpoint of the last bin is the maximum of `time_variable`.

## Value

A [`BinnedData`](@ref) object.
"""
function bin_time(data::DataFrame; time_variable = "date", bin_resolution = 10)
    x = deepcopy(data)

    # cut points
    cuts = range(minimum(x[!, time_variable]); stop = maximum(x[!, time_variable]), length = bin_resolution + 1)

    # midpoints
    mids = [cuts[i] + (cuts[i+1] - cuts[i])/2 for i in 1:bin_resolution]

    # bin data
    x.timebin = cut(x[!, time_variable],
                    cuts;
                    extend = true)

    # convert date, which is now a CategoricalArray, into Int
    x.timebin = levelcode.(x.timebin)

    BinnedData(x, bin_resolution, mids)
end
