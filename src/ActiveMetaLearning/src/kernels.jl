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
    K_XX = KernelFunctions.kernelmatrix(dist.k, X, obsdim=1)
    αᵀKα = α'*K_XX*α

    Y = q.xs
    β = reshape(q.p, :, 1)
    K_YY = KernelFunctions.kernelmatrix(dist.k, Y, obsdim=1)
    βᵀKβ = β'*K_YY*β

    K_XY = KernelFunctions.kernelmatrix(dist.k, X, Y, obsdim=1)
    αᵀKβ = α'*K_XY*β
    mmd² = (αᵀKα + βᵀKβ - 2αᵀKβ)[1, 1]
    @assert mmd² + 10 * eps(eltype(mmd²)) >= 0.0, "The value of mmd² should be non-negative, but is $(mmd²)"
    mmd = .√(mmd² + 10 * eps(eltype(mmd²)))
    return mmd
end

"""
Calculate the mmd pairwise between two vectors `ps` and `qs` of
`CategoricalDistribution`'s. We do this efficiently by using the fact that,
letting K^{i, j} being the kernel matrix between `p[i].xs` and `q[j].xs` and denoting
`p[i].p` by `σ^i` and `q[j].p` by `β^j`, we can write
    mmd²_grammian_{i, j} = α^i'*K^{i, i}*α^i + β^j'*K^{j, j}*β^j - 2α^i'*K^{i, j}*β^j
and see that the first and second term can be broadcasted over the columns and rows respectively.

Return the mmd_grammian (square root of mmd²_grammian).
"""
function (dist::MaximumMeanDiscrepancy)(ps::Vector{CategoricalDistribution}, qs::Vector{CategoricalDistribution})
    p_len = length(ps)
    q_len = length(qs)
    mmd²_grammian = zeros(eltype(ps[1].xs), p_len, q_len)
    # Broadcast over columns
    for i ∈ range(1, stop=length(ps))
        X = ps[i].xs
        α = reshape(ps[i].p, :, 1)
        K_XX = KernelFunctions.kernelmatrix(dist.k, X, obsdim=1)
        αᵀKα = α'*K_XX*α
        for j ∈ range(1, stop=length(qs))
            mmd²_grammian[i, j] += αᵀKα[1, 1]
        end
    end
    # Broadcast over rows
    for j ∈ range(1, stop=length(qs))
        Y = qs[j].xs
        β = reshape(qs[j].p, :, 1)
        K_YY = KernelFunctions.kernelmatrix(dist.k, Y, obsdim=1)
        βᵀKβ = β'*K_YY*β
        for i ∈ range(1, stop=length(ps))
            mmd²_grammian[i, j] += βᵀKβ[1, 1]
        end
    end
    # Calculate α'*K_XY*β
    for j ∈ range(1, stop=length(qs))
        for i ∈ range(1, stop=length(ps))
            X = ps[i].xs
            α = reshape(ps[i].p, :, 1)
            Y = qs[j].xs
            β = reshape(qs[j].p, :, 1)
            K_XY = KernelFunctions.kernelmatrix(dist.k, X, Y; obsdim=1)
            αᵀKβ = α'*K_XY*β
            mmd²_grammian[i, j] -= 2αᵀKβ[1, 1]
        end
    end
    @assert all(mmd²_grammian .+ 10 * eps(eltype(mmd²_grammian)) .>= 0.0) "The values of the grammian should be non-negative but the smallest value is $(minimum(mmd²_grammian))"
    mmd_grammian = .√(mmd²_grammian .+ 10 * eps(eltype(mmd²_grammian)))
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
    # In future could speed up by leveraging symmetry of
    # Grammian and reuse of terms in MMD calculation
    return dist(ps, ps)
end

####################################################
# Kernel on distributions (conv. comb. of diracs)  #
# we follow KernelFunctions to the extent possible #
####################################################

"""
The MMDKernel calculates kernels on CategoricalDistribution.

Would want to use 2 kernels, but will use the base_κ as the original used in MMD
and then the mapping for the second one
"""
struct MMDKernel{K<:Kernel, V<:Kernel}
    base_k::K
    higher_k::V
end

# Will have to hardcode this until there is a way to use Distances.jl on the CategoricalDistribution
function KernelFunctions.kernelmatrix(k::MMDKernel{<:Kernel, SqExponentialKernel{T, Tr}}, ps::Vector{CategoricalDistribution}, qs::Vector{CategoricalDistribution}) where {T, Tr<:Transform}
    mmd = MMD(k.base_k)
    mmd_grammian = mmd(ps, qs)
    # Require transform to be scalar Kernel.ScaleTransform
    ρ²mmd²_grammian = transform(k.higher_k.transform, mmd_grammian).^2
    return exp.(-ρ²mmd²_grammian)
end
