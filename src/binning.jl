"""
    bin_time(x::DataFrame; time_variable = "date", bin_resolution = 10)

Bin the time variable in a `DataFrame` into `bin_resolution` equally
spaced bins. Returns a `BinnedCorpus`.
"""
function bin_time(x::DataFrame; time_variable = "date", bin_resolution = 10)
    # start point of time series in real time
    sp = minimum(x[!, time_variable])

    # bin data
    x.date = cut(x[!, time_variable],
                 minimum(x[!, time_variable]):bin_resolution:maximum(x[!, time_variable]);
                 labels = (f, t, i; leftclosed, rightclosed, sigdigits) -> f + (f-t)/2,
                 extend = true)

    # convert date, which is now a CategoricalArray, into Int
    x.date = levelcode.(x.date)
    x.date = sp .+ bin_resolution .* x.date

    BinnedCorpus(x, bin_resolution)
end

