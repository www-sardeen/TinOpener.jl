function transfer_entropy(x::Vector{Float64}, y::Vector{Float64};
        nbins = 2,
        base = 2,
        bidirectional = false)
    discr = CodifyVariables(TransferOperator(RectangularBinning(nbins, true)))
    est = EntropyDecomposition(TEShannon(; base = base), PlugIn(Shannon()), discr)
    #discr = CodifyVariables(ValueHistogram(nbins))
    #est = JointProbabilities(TEShannon(; base = base), discr)

    if bidirectional
        return association(est, x, y), association(est, y, x)
    else
        return association(est, x, y)
    end
end


function TE_permutation_test(x::Vector{Float64}, y::Vector{Float64};
        reps = 1000,
        alpha = 0.05,
        nbins = 2,
        base = 2,
        bidirectional = false,
        return_distribution = true)
    # helper function to carry out test in one direction
    function do_one(x::Vector{Float64}, y::Vector{Float64};
            reps, alpha, nbins, base)
        empirical = transfer_entropy(x, y; nbins = nbins, base = base, bidirectional = false)

        permuted = [transfer_entropy(shuffle(x), y; nbins = nbins, base = base, bidirectional = false) for i in 1:reps]

        threshold = Statistics.quantile(permuted, 1 - alpha)

        significant = empirical > threshold ? true : false

        pvalue = sum(permuted .> empirical) ./ reps

        distro = return_distribution ? permuted : nothing

        return TEPermutationTestResult(empirical, threshold, pvalue, alpha, significant, distro)
    end

    if bidirectional
        return do_one(x, y; reps = reps, alpha = alpha, nbins = nbins, base = base), do_one(y, x; reps = reps, alpha = alpha, nbins = nbins, base = base)
    else
        return do_one(x, y; reps = reps, alpha = alpha, nbins = nbins, base = base)
    end
end
