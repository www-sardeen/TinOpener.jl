function shuffle(x::BinnedTimeSeries)
    data = copy(x.data)

    data.value = TimeseriesSurrogates.surrogate(data.value, TimeseriesSurrogates.RandomShuffle())

    BinnedTimeSeries(data, x.bin_resolution, x.midpoints)
end

function shuffle(x::Vector{Float64})
    y = copy(x)

    y = TimeseriesSurrogates.surrogate(y, TimeseriesSurrogates.RandomShuffle())

    return y
end
