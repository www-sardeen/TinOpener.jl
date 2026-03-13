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


