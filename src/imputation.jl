"""
    impute(x::BinnedTimeSeries; nnsize = 2)

Carry out nearest-neighbour imputation (with neighbourhood size `nnsize`)
over a `BinnedTimeSeries`.

Return value is a `BinnedTimeSeries` with `missing` values replaced
by imputed values.
"""
function impute_kalman(x::BinnedTimeSeries)
    data = copy(x.data)

    data.value = float.(RCall.R"imputeTS::na_kalman($(data.value), smooth=TRUE)")

    BinnedTimeSeries(data, x.bin_resolution, true)
end


#=
#OLD version using TSML which we are now giving up in favour
#of RCalling R's imputeTS package (original motivation for this
#was a package version clash in Julia, but it is also the case
#that imputeTS provides more methods, such as a Kalman imputation)
#
function impute(x::BinnedTimeSeries; nnsize = 2)
    data = x.data[:, [:date, :value]]

    # date from Int to Date
    data.date = Dates.Date.(data.date)

    # impute missing values
    datevalnner = TSML.DateValNNer(Dict(:dateinterval => Dates.Year(x.bin_resolution),
                                        :nnsize => nnsize))
    data = TSML.fit_transform!(datevalnner, data)
    data.n = x.data.n
    data.Date = Dates.year.(data.Date)

    BinnedTimeSeries(data, x.bin_resolution, true)
end
=#
