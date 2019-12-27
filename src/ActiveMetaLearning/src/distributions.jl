# using Distributions

############################################
# Distribution struct (convex combination) #
############################################
# Note, could use Distributions.jl but would have to rewrite things
# For now just define an empirical distribution directly
"""
    CategoricalDistribution(xs, p)

Categorical distribution, `xs` is a matrix
and `p` is the corresponding probability vector as an array.
"""
struct CategoricalDistribution
    xs::Matrix{Float64}
    p::Vector{Float64}
    function CategoricalDistribution(xs::Matrix{Float64}, p::Vector{Float64})
        @assert isapprox(sum(p), 1.0; atol=1e-7, rtol=0)
        @assert all(p .> 0.0)
        @assert size(xs, 1) == size(p)[1]
        return new(xs, p)
    end
end
CategoricalDistribution(xs) = CategoricalDistribution(xs, ones(size(xs, 1)) ./ size(xs, 1))
CategoricalDistribution() = CategoricalDistribution(reshape([0.0], 1, 1), [1.0]) # Empty initialisation is dirac over 0.0
function DiracDistribution(x::Matrix{Float64})
    @assert size(x, 1) == 1
    return CategoricalDistribution(x, [1.0])
end
const EmpiricalDistribution(xs) = CategoricalDistribution(xs)
