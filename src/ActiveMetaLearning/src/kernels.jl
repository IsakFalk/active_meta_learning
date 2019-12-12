using KernelFunctions
using Distances
using LinearAlgebra

include("distributions.jl")

"""
Implements the Maxium Mean Discrepancy distance (MMD).

Note that technically the MMD is not a metric unless the kernel supplied
is characteristic, but in general a pseudometric in that MMD(p, q) = 0 does
not imply that p == q. We do not use Distances.jl for this as we need to pass
information about the points and not just the coefficients.

function evaluate(dist::MaximumMeanDiscrepancy, p::CategoricalDistribution, q::CategoricalDistribution)

Based upon the [bregman distance code](https://github.com/JuliaStats/Distances.jl/blob/master/src/bregman.jl).
"""
struct MaximumMeanDiscrepancy{K<:Kernel} #<: Metric
    k::K
end
const MMD = MaximumMeanDiscrepancy

# Evaluation function
function (dist::MaximumMeanDiscrepancy)(p::CategoricalDistribution, q::CategoricalDistribution)
    X = p.xs
    α = reshape(p.p, :, 1)
    K_XX = kernelmatrix(dist.k, X, obsdim=1)
    αᵀKα = α'*K_XX*α

    Y = q.xs
    β = reshape(q.p, :, 1)
    K_YY = kernelmatrix(dist.k, Y, obsdim=1)
    βᵀKβ = β'*K_YY*β

    K_XY = kernelmatrix(dist.k, X, Y, obsdim=1)
    αᵀKβ = α'*K_XY*β

    # Return the MMD.
    return (αᵀKα + βᵀKβ - 2αᵀKβ)[1]
end

"""
Calculate the mmd pairwise between two vectors `ps` and `qs` of
`CategoricalDistribution`'s. We do this efficiently by using the fact that,
letting K^{i, j} being the kernel matrix between `p[i].xs` and `q[j].xs` and denoting
`p[i].p` by `σ^i` and `q[j].p` by `β^j`, we can write
    mmd_grammian_{i, j} = α^i'*K^{i, i}*α^i + β^j'*K^{j, j}*β^j - 2α^i'*K^{i, j}*β^j
and see that the first and second term can be broadcasted over the columns and rows respectively.
"""
function (dist::MaximumMeanDiscrepancy)(ps::Vector{CategoricalDistribution}, qs::Vector{CategoricalDistribution})
    p_len = length(ps)
    q_len = length(qs)
    mmd_grammian = zero(eltype(ps[1].xs), p_len, q_len)
    # Broadcast over columns
    for i ∈ range(1, stop=length(ps))
        X = ps[i].xs
        α = reshape(ps[i].p, :, 1)
        K_XX = kernelmatrix(dist.k, X, obsdim=1)
        αᵀKα = α'*K_XX*α
        for j ∈ range(1, stop=length(qs))
            mmd_grammian[i, j] += αᵀKα
        end
    end
    # Broadcast over rows
    for j ∈ range(1, stop=length(qs))
        Y = qs[j].xs
        β = reshape(qs[j].p, :, 1)
        K_YY = kernelmatrix(dist.k, Y, obsdim=1)
        βᵀKβ = β'*K_YY*β
        for i ∈ range(1, stop=length(ps))
            mmd_grammian[i, j] += βᵀKβ
        end
    end
    # Calculate α'*K_XY*β
    for j ∈ range(1, stop=length(qs))
        for i ∈ range(1, stop=length(ps))
            X = ps[i].xs
            α = reshape(ps[i].p, :, 1)
            Y = qs[j].xs
            β = reshape(qs[j].p, :, 1)
            K_XY = kernelmatrix(dist.k, X, Y, obsdim=1)
            αᵀKβ = α'*K_XY*β
            mmd_grammian[i, j] -= 2αᵀKβ
        end
    end
    return mmd_grammian
end

"""
Calculate the mmd pairwise between all entries of `ps` os
`CategoricalDistribution`'s. We do this efficiently by using the fact that,
letting K^{i, j} being the kernel matrix between `p[i].xs` and `p[j].xs` and denoting
`p[i].p` by `σ^i`,
    mmd_grammian_{i, j} = α^i'*K^{i, i}*α^i + α^j'*K^{j, j}*α^j - 2α^i'*K^{i, j}*σ^j
and see that the first and second term can be broadcasted over the columns and rows respectively.
"""
function (dist::MaximumMeanDiscrepancy)(ps::Vector{CategoricalDistribution})
    p_len = length(ps)
    mmd_grammian = zero(eltype(ps[1].xs), p_len, p_len)
    # Broadcast over columns
    for j ∈ range(2, stop=length(ps))
        for i ∈ range(2, stop=j)
            X = ps[i].xs
            α = reshape(ps[i].p, :, 1)
            K_XX = kernelmatrix(dist.k, X, obsdim=1)
            αᵀKα = α'*K_XX*α
            for j ∈ range(1, stop=length(qs))
                mmd_grammian[i, j] += αᵀKα
            end
        end
    end
    # Broadcast over rows
    for j ∈ range(1, stop=length(qs))
        Y = qs[j].xs
        β = reshape(qs[j].p, :, 1)
        K_YY = kernelmatrix(dist.k, Y, obsdim=1)
        βᵀKβ = β'*K_YY*β
        for i ∈ range(1, stop=length(ps))
            mmd_grammian[i, j] += βᵀKβ
        end
    end
    # Calculate α'*K_XY*β
    for j ∈ range(1, stop=length(qs))
        for i ∈ range(1, stop=length(ps))
            X = ps[i].xs
            α = reshape(ps[i].p, :, 1)
            Y = qs[j].xs
            β = reshape(qs[j].p, :, 1)
            K_XY = kernelmatrix(dist.k, X, Y, obsdim=1)
            αᵀKβ = α'*K_XY*β
            mmd_grammian[i, j] -= 2αᵀKβ
        end
    end
    return mmd_grammian
end
