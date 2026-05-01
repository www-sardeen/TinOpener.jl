"""
    logistic(x::Float64)

Return the logistic transformation of `x`,
``f(x) = \\frac{1}{1 + \\exp (-x)}``
"""
logistic(x::Float64) = 1 / (1 + exp(-x))

logistic(x::Missing) = missing


"""
    logit(x::Float64)

Return the logit transform of `x`,
``f(x) = \\log\\left(\\frac{x}{1 - x}\\right)``
"""
logit(x::Float64) = log(x / (1 - x))

logit(x::Missing) = missing


"""
    aic(k::Int, logL:Float64)

Akaike information criterion for model with log-likelihood `logL` and number
of model parameters `k`, i.e. ``\\textnormal{AIC} = 2k - \\log L``
where ``L`` is the likelihood.
"""
aic(k::Int, logL::Float64) = 2*k - 2*logL


"""
    aicc(k::Int, logL::Float64, n::Int)

Small-sample corrected Akaike information criterion for model with log-likelihood `logL`, number
of model parameters `k` and sample size `n`, i.e. ``\\textnormal{AICc} = 2k - \\log L + \\frac{2k^2 + 2k}{n - k - 1}``
where ``L`` is the likelihood.
"""
aicc(k::Int, logL::Float64, n::Int) = 2*k - 2*logL + (2*k^2 + 2*k)/(n - k - 1)


"""
    impute_last!(x)

Replaces each `missing` value in vector `x` with the most recent
non-missing value in that vector, in-place.
"""
function impute_last!(x)
    for i in 1:length(x)
        if ismissing(x[i])
            x[i] = x[i-1]
        end
    end
end


"""
    impute_last!(x::BinnedTimeSeries, column::Symbol)

Replaces each `missing` value in specified `column` of a `BinnedTimeSeries`
with the most recent non-missing value (in-place).
"""
function impute_last!(x::BinnedTimeSeries, column::Symbol)
    impute_last!(x.data[column])
end


