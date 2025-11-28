mutable struct BinnedCorpus
    data::DataFrame
    bin_resolution::Int
end

mutable struct BinnedTimeSeries
    data::DataFrame
    bin_resolution::Int
    imputed::Bool
end
