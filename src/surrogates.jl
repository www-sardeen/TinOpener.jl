"""
    shuffle(x::BinnedTimeSeries)

Return a random permutation of a [`BinnedTimeSeries`](@ref). Mainly
useful for surrogate testing purposes.
"""
function shuffle(x::BinnedTimeSeries)
    data = copy(x.data)

    data.value = TimeseriesSurrogates.surrogate(data.value, TimeseriesSurrogates.RandomShuffle())

    BinnedTimeSeries(data, x.bin_resolution, x.midpoints)
end


"""
    shuffle(x::Vector{Float64})

Return a random permutation of an array of floats. Mainly
useful for surrogate testing purposes.
"""
function shuffle(x::Vector{Float64})
    y = copy(x)

    y = TimeseriesSurrogates.surrogate(y, TimeseriesSurrogates.RandomShuffle())

    return y
end
