"""
    mapply(fun, vars)

Multidimensional apply: apply function `fun` to each combination
of the variable values in `vars`, a `NamedTuple`.

## Value

Returns a named tuple with fields:

- `grid`: grid of values, a `DataFrame`
- `result`: vector of results, i.e. the output of `fun` when applied to each row of `grid`

## Example

```julia
using Distributed

addprocs(6)

@everywhere myfun(x) = sum(x)

mapply(myfun, (var1 = 1:3, var2 = 1:10, var3 = 10:20))
```
"""
function mapply(fun,
                vars)
    grid = DataFrames.allcombinations(DataFrame; vars...)
    result = pmap(x -> fun(x), eachrow(grid))
    
    return (grid = grid, result = result)
end


# older version, not really needed anymore, retained for
# archeological reasons
#
function mapply_OLD(fun,
                vars)
    grid = DataFrames.allcombinations(DataFrame; vars...)
    
    result = @sync @distributed (vcat) for r in eachrow(grid)
        df = DataFrame(r)
        df.result .= fun(r)
        df
    end

    result
end
