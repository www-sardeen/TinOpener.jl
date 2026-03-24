abstract type AbstractBinnedData end


"""
    BinnedData

A data structure containing a time-binned dataframe. Fields:

- `data`: the dataframe
- `bin_resolution`: binning resolution (see [`bin_time`](@ref))
- `midpoints`: midpoints of the time bins on the original time scale
"""
mutable struct BinnedData <: AbstractBinnedData
    data::DataFrame
    bin_resolution::Int
    midpoints::Vector{Float64}
end


# find the (first) index of x in vector y
#
arg(x, y) = findfirst(==(x), y)


mutable struct SqueezedData <: AbstractBinnedData
    data::DataFrame
    bin_resolution::Int
    midpoints::Vector{Float64}
    sampling_intervals::Vector{Float64}
end


function squeeze_time(data::DataFrame; time_variable = "date")
    # times at which data have been "sampled"
    sampling_times = @pipe sort(unique(data[!, time_variable])) |> convert.(Float64, _)

    # sampling intervals, i.e. time between two consecutive
    # sampling points
    sampling_intervals = vcat(0.0, @pipe [sampling_times[i] - sampling_times[i-1] for i in 2:length(sampling_times)] |> convert.(Float64, _))

    # add sampling times to dataframe
    x = transform(data, time_variable => (d -> arg.(d, [sampling_times])) => :timebin)

    # return
    SqueezedData(x, length(sampling_times), sampling_times, sampling_intervals)
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


function bin_time_natural(data::DataFrame;
        time_variable = "date")
    x = deepcopy(data)

    x.timebin = x[!, time_variable] .- minimum(x[!, time_variable])

    BinnedData(x, 1, minimum(x[!, time_variable]):1:maximum(x[!, time_variable]))
end


function bin_time(data::DataFrame,
        bin_length::Int;
        bin_offset = 0,
        time_variable = "date")
    # make a deep copy of dataframe
    x = deepcopy(data)

    # left and right values of time variable
    left, right = extrema(x[!, time_variable])

    # delta between left and right
    delta = right - left

    # delta between left point of leftmost bin, taking
    # offset into account, and right point of data
    delta_overall = delta + bin_offset

    # how many bins we need to cover all data
    nbins = round(Int, ceil(delta_overall / bin_length)) + 1

    # cut points
    cuts = range(; start = left - bin_offset, step = bin_length, length = nbins)

    # midpoints
    mids = [cuts[i] + (cuts[i+1] - cuts[i])/2 for i in 1:(nbins - 1)]

    # bin data
    x.timebin = cut(x[!, time_variable], cuts; extend = true)

    # convert date, which is now a CategoricalArray, into Int
    x.timebin = levelcode.(x.timebin)

    BinnedData(x, nbins, mids)
end
