"""
    impute(x::BinnedTimeSeries; nnsize = 2)

Carry out nearest-neighbour imputation (with neighbourhood size `nnsize`)
over a `BinnedTimeSeries`.

Return value is a `BinnedTimeSeries` with `missing` values replaced
by imputed values.
"""
function impute(x::BinnedTimeSeries; nnsize = 2)
    data = x.data[:, [:date, :value]]

    # date from Int to Date
    data.date = Date.(data.date)

    # impute missing values
    datevalnner = TSML.DateValNNer(Dict(:dateinterval => Dates.Year(x.bin_resolution),
                                        :nnsize => nnsize))
    data = TSML.fit_transform!(datevalnner, data)
    data.n = x.data.n
    data.Date = year.(data.Date)

    BinnedTimeSeries(data, x.bin_resolution, true)
end

