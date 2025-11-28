"""
    bootstrap(x::DataFrame)

Bootstrap a `DataFrame`, i.e. sample its rows with replacement.
"""
function bootstrap(x::DataFrame)
    x[StatsBase.sample(1:nrow(x), nrow(x); replace = true), :]
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
