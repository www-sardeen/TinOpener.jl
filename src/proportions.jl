"""
    categorical_to_proportion(x::BinnedCorpus, variable, ref_level;
        expand_missing = true, shuffle = false)

Transform a categorical variable into its proportion, calculated per
time bin.

The `variable` is first transformed into a vector of ones and zeroes;
whenever the value coincides with `ref_level`, we replace the value with
a one, otherwise we replace it with a zero. The proportion of `ref_level`
per time bin is then calculated as the mean over this vector of ones and
zeroes.

The output is a `BinnedTimeSeries`.
"""
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

