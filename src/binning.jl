"""
    bin_time(x::DataFrame; time_variable = "date", bin_resolution = 10)

Bin the time variable in a `DataFrame` into `bin_resolution` equally
spaced bins. Returns a `BinnedData` object.
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

