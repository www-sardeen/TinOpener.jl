"""
    logistic(x::Float64)

Return the logistic transformation of `x`.
"""
logistic(x::Float64) = 1 / (1 + exp(-x))

logistic(x::Missing) = missing


"""
    logit(x::Float64)

Return the logit transform of `x`.
"""
logit(x::Float64) = log(x / (1 - x))

logit(x::Missing) = missing


"""
    aic(k::Int, ll:Float64)

Akaike information for model with log-likelihood `ll` and number
of model parameters `k`.
"""
aic(k::Int, ll::Float64) = 2*k - 2*ll


