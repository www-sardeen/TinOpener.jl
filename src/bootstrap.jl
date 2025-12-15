"""
    bootstrap(x::DataFrame)

Bootstrap a `DataFrame`, i.e. sample its rows with replacement.
"""
function bootstrap(x::DataFrame)
    d = Distributions.DiscreteUniform(1, nrow(x))
    x[rand(d, nrow(x)), :]
end


"""
    bootstrap(x::SubDataFrame)

Bootstrap a `SubDataFrame`, i.e. sample its rows with replacement.
"""
function bootstrap(x::SubDataFrame)
    d = Distributions.DiscreteUniform(1, nrow(x))
    x[rand(d, nrow(x)), :]
end


"""
    bootstrap(x::GroupedDataFrame{DataFrame})

Bootstrap a `GroupedDataFrame`, i.e. separately call `bootstrap` on
each of its subdataframes.
"""
function bootstrap(x::GroupedDataFrame{DataFrame})
    for y in x
        bootstrap(y)
    end

    x
end
