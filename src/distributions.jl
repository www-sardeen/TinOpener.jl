"""
    BiBeta

A bivariate beta distribution.

The Olkin--Trikalinos construction for a bivariate beta distribution,
with parameters `α11`, `α10`, `α01` and `α00`.

# Details

The construction is based on a Dirichlet distribution.
Suppose ``(U_{11}, U_{10}, U_{01}, U_{00}) \\sim \\textnormal{Dirichlet}(\\alpha)`` 
where ``\\alpha = (\\alpha_{11}, \\alpha_{10}, \\alpha_{01}, \\alpha_{00})``.
Define ``X = U_{11} + U_{10}`` and ``Y = U_{11} + U_{01}``.
Then the joint density of ``X`` and ``Y`` is

```math
f(x,y) = B(\\alpha)^{-1} \\int_{\\Omega} u_{11}^{\\alpha_{11} - 1}
(x - u_{11})^{\\alpha_{10} - 1} (y - u_{11})^{\\alpha_{01} - 1}
(1 - x - y + u_{11})^{\\alpha_{00} - 1} du_{11}
```

with ``\\Omega = \\{u_{11} : \\max(0, x+y-1) < u_{11} < \\min(x,y)\\}``
and ``B(\\alpha)`` the generalized beta function
``B(\\alpha) = \\Gamma(\\alpha_{11}) \\Gamma(\\alpha_{10}) \\Gamma(\\alpha_{01}) \\Gamma(\\alpha_{00}) / \\Gamma(\\alpha_{11} + \\alpha_{10} + \\alpha_{01} + \\alpha_{00})``.

## Reference

Olkin, I. & Trikalinos, T. A. (2015) Constructions for a bivariate
beta distributions. *Statistics and Probability Letters*, 96, 54--60.
http://dx.doi.org/10.1016/j.spl.2014.09.013.
""" 
mutable struct BiBeta
    α11::Float64
    α10::Float64
    α01::Float64
    α00::Float64
end


"""
    get_parameters(d::BiBeta)

Return the parameters of a `BiBeta` distribution as a vector.
"""
get_parameters(d::BiBeta) = [d.α11, d.α10, d.α01, d.α00]


# shorthand for the logarithmic gamma function
#
lg(x::Float64) = SpecialFunctions.loggamma(x)


# logarithm of the inverse of the generalized Beta function, i.e.
# the normalization factor for the BiBeta density
# 
liB(d::BiBeta) = lg(d.α11 + d.α10 + d.α01 + d.α00) - lg(d.α11) - lg(d.α10) - lg(d.α01) - lg(d.α00)


"""
    logpdf(d::BiBeta, x::Vector{Float64})

Logarithmic density of the bivariate beta distribution `d`,
evaluated at point `x`.

The integral in the density is solved numerically using FIXME.
"""
function logpdf(d::BiBeta, x::Vector{Float64})
    f(u, p) = u^(d.α11 - 1) * (x[1] - u)^(d.α10 - 1) * (x[2] - u)^(d.α01 - 1) * (1 - x[1] - x[2] + u)^(d.α00 - 1)

    domain = (maximum([0.0, x[1] + x[2] - 1.0]), minimum(x))

    prob = Integrals.IntegralProblem(f, domain)

    sol = solve(prob, Integrals.HCubatureJL(); reltol = 1e-3, abstol = 1e-3)

    return liB(d) + log(sol.u)
end


"""
    pdf(d::BiBeta, x::Vector{Float64})

Density of the bivariate beta distribution `d`, evaluated at point `x`.

The integral in the density is solved numerically using FIXME.
"""
pdf(d::BiBeta, x::Vector{Float64}) = exp(logpdf(d, x))


"""
    momentmatch(μ1, μ2, var1, var2, ρ)

Find a bivariate beta distribution (`BiBeta`), i.e. its parameters, by
moment-matching.

# Arguments

- `μ1`: mean of first covariate
- `μ2`: mean of second covariate
- `var1`: variance of first covariate
- `var2`: variance of second covariate
- `ρ`: correlation between covariates

# Value

A `NamedTuple` with the following fields:

- `d`: distribution found
- `sol`: solution object from numerical solver

# Details

The best-fitting parameter vector ``\\alpha`` is found by numerical
optimization as per the procedure outlined by Olkin & Trikalinos (2015).
The Nelder--Mead algorithm is used.

# Reference

Olkin, I. & Trikalinos, T. A. (2015) Constructions for a bivariate
beta distributions. *Statistics and Probability Letters*, 96, 54--60.
http://dx.doi.org/10.1016/j.spl.2014.09.013.
"""
function momentmatch(μ1, μ2, var1, var2, ρ)
    mu = μ1
    nu = μ2
    V1 = var1
    V2 = var2
    rho = ρ

    function objective(u)
        a11 = exp(u[1])
        a10 = exp(u[2])
        a01 = exp(u[3])
        a00 = exp(u[4])
        M = a11 + a10 + a01 + a00

        return (mu - (a11 + a10)/M)^2 + (nu - (a11 + a01)/M)^2 + (V1 - ((a11 + a10)*(a00 + a01))/((M^2*(M+1))))^2 + (V2 - ((a11 + a01)*(a00 + a10))/(M^2*(M+1)))^2 + (rho - (a11*a00 - a10*a01)/sqrt((a11 + a10)*(a11*a01)*(a01*a00)*(a00*a10)))^2
    end

    optf = OptimizationFunction((u, p) -> objective(u))

    # initial guess
    u0 = 0.01 .* rand(4)

    # define optimization problem
    prob = OptimizationProblem(optf, u0, nothing)

    # obtain solution
    sol = solve(prob, OptimizationOptimJL.NelderMead())

    return (d = BiBeta(exp.(sol.u)...), sol = sol)
end



