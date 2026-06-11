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


# rising factorial
#
rf(x::Float64, n::Int64) = SpecialFunctions.gamma(x + n) / SpecialFunctions.gamma(x)


"""
    appell_F1(a, b1, b2, c, x, y; tol = 1e-3)

Compute the Appell hypergeometric series ``F_1`` of two variables, `x`
and `y`, with parameters `a`, `b1` and `b2`.

The implementation assumes that the series is convergent, i.e.
that ``|x| < 1`` and ``|y| < 1``. If this is not the case, the
function exits with an error.

The infinite series is computed up to a tolerance of `tol`, with
a default value of `1e-3`. In other words, the computation is
terminated once two consecutive members of the series are with
`tol` of each other.
"""
function appell_F1(a::Float64,
        b1::Float64,
        b2::Float64,
        c::Float64,
        x::Float64,
        y::Float64;
        tol = 1e-6)
    if abs(x) >= 1 || abs(y) >= 1
        throw("need abs(x) < 1 and abs(y) < 1 for series to converge")
    end

    F1 = 0.0

    error = tol + 1.0

    m = 0

    while (error > tol)
        oldF1 = F1

        for n in 0:m
            F1 += ((rf(a, m+n) * rf(b1, m) * rf(b2, n)) / (rf(c, m+n) * factorial(m) * factorial(n))) * x^m * y^n
        end

        error = abs(oldF1 - F1)

        m += 1
    end

    return F1
end


"""
    lrtest(mod_null::AbstractModelFit, mod_alt::AbstractModelFit)

Likelihood ratio test.
"""
function lrtest(mod_null::AbstractModelFit, mod_alt::AbstractModelFit)
    statistic = -2 * (mod_null.ll - mod_alt.ll)
    df = mod_alt.npar - mod_null.npar
    pvalue = 1 - Distributions.cdf(Distributions.Chisq(df), statistic)

    return (statistic = statistic, df = df, pvalue = pvalue)
end


"""
    conditional_entropy(M; b = 2)

Computes the conditional entropy for a matrix or matrix-like object
`M`, taking the rows as the conditioning and the columns as the
conditioned variable. Base of the logarithm (determining the unit)
is given by `b`; the default is `b = 2` i.s. bits.
"""
function conditional_entropy(M; b = 2)
    N = sum(M)

    fun(x) = log(x) / log(b)

    sum([M[x,y] == 0.0 ? 0.0 : -(M[x,y]/N) * (fun(M[x,y]/N) - fun(sum(M[x,:])/N)) for x in 1:size(M)[1], y in 1:size(M)[2]])
end


function mutual_information(M; b = 2)
    N = sum(M)

    fun(x) = log(x) / log(b)

    sum([M[x,y] == 0.0 ? 0.0 : (M[x,y]/N) * (fun(M[x,y]/N) - fun(sum(M[x,:])/N) - fun(sum(M[:,y]/N))) for x in 1:size(M)[1], y in 1:size(M)[2]])
end


