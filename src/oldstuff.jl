using CategoricalArrays
using DataFrames
using Dates
using Pipe
using StatsBase
using TSML


mutable struct BinnedCorpus
    data::DataFrame
    bin_resolution::Int
end


mutable struct BinnedTimeSeries
    data::DataFrame
    bin_resolution::Int
    imputed::Bool
end


function bootstrap(x::DataFrame)
    x[StatsBase.sample(1:nrow(x), nrow(x); replace = true), :]
end

function bootstrap(x::GroupedDataFrame{DataFrame})
    for y in x
        bootstrap(y)
    end

    x
end


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

function categorical_to_proportion(x::BinnedCorpus, variable, ref_level;
        expand_missing = true, shuffle = false)
    # make response_variable numeric
    x.data.resp = ifelse.(x.data[!, variable] .== ref_level, 1, 0)

    # obtain means per dates
    if shuffle
        x.data = @pipe groupby(x.data, :date) |> bootstrap |> combine(_, bootstrap)
    end

    x.data = @pipe groupby(x.data, :date) |> combine(_, :resp => mean => :value, nrow => :n)

    # when a date has no data, flag this up explicitly as missing
    if expand_missing
        x.data = @pipe leftjoin(DataFrame(date = minimum(data.date):x.bin_resolution:maximum(data.date)), x.data; on = :date) |> sort(_, :date)
    end

    BinnedTimeSeries(x.data, x.bin_resolution, false)
end

function impute(x::BinnedTimeSeries; nnsize = 2)
    data = x.data[:, [:date, :value]]

    # date from Int to Date
    data.date = Date.(data.date)

    # impute missing values
    datevalnner = DateValNNer(Dict(:dateinterval => Dates.Year(x.bin_resolution),
                                   :nnsize => nnsize))
    data = fit_transform!(datevalnner, data)
    data.n = x.data.n
    data.Date = year.(data.Date)

    BinnedTimeSeries(data, x.bin_resolution, true)
end


function myfun(x)
    @pipe bin_time(x; time_variable = "mean_date", bin_resolution = 10) |> categorical_to_proportion(_, :ambig_RT, "ambiguous"; expand_missing = true, shuffle = true) |> impute
end
