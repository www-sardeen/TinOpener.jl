"""
    convolve(x::Vector{Float64}, t::Int, K::Function, S::Int)

Obtain a discrete convolution of vector `x` with convolution kernel `K`
at time/index `t`, up to `S` times in the past.

In other words, compute the quantity

```math
(K \\star x)_t = \\sum_{s=0}^S K(s) x(t-s).
```
"""
function convolve(x::Vector{Float64}, t::Int, K::Function, S::Int;
        normalize = false)
    normalizer = normalize ? sum(K.(0:S)) : 1.0
    
    return sum([K(s) * x[t - s] for s in 0:S]) / normalizer
end


"""
    weak_kernel(s; T = 1.0)

The "weak" (exponentially decaying) convolution kernel

```math
K(s) = \\frac{1}{T} e^{-s/T}.
```

# Reference

Cushing FIXME
"""
weak_kernel(s::Float64; T = 1.0) = (1/T) * exp(-s/T)
weak_kernel(s::Int64; T = 1.0) = (1/T) * exp(-s/T)


"""
    strong_kernel(s; T = 1.0)

The "strong" convolution kernel

```math
K(s) = \\frac{s}{T^2} e^{-s/T}.
```

This has maximum at ``s = T``.

# Reference

Cushing FIXME
"""
strong_kernel(s::Float64; T = 1.0) = (s/T^2) * exp(-s/T)
strong_kernel(s::Int64; T = 1.0) = (s/T^2) * exp(-s/T)


"""
    zc_kernel(s; T = 1.0, U = 2.0)

The "zero-crossing" kernel

```math
K(s) = (U - s) \\frac{s}{T^2} e^{-s/T}.
```

"""
zc_kernel(s::Float64; T = 1.0, U = 2.0) = (U - s) * (s/T^2) * exp(-s/T)
zc_kernel(s::Int64; T = 1.0, U = 2.0) = (U - s) * (s/T^2) * exp(-s/T)


"""
    bzc_kernel(s; T = 1.0)

The "balanced zero-crossing" kernel

```math
K(s) = (2T - s) \\frac{s}{T^2} e^{-s/T}.
```

This places equal mass on each side of the crossover point.
"""
bzc_kernel(s::Float64; T = 1.0) = (2*T - s) * (s/T^2) * exp(-s/T)
bzc_kernel(s::Int64; T = 1.0) = (2*T - s) * (s/T^2) * exp(-s/T)



