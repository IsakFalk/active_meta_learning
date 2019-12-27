module ActiveMetaLearning

include("tasks.jl")
export MetaLearningEnvironment, MetaLearningRegEnvironment, MetaLearningClassEnvironment
export sine_wave_environment, sine_wave_linear_mixture_environment
export common_mean_ex1_environment, common_mean_ex2_environment
export low_dim_lin_reg_environment, isotropic_lin_reg_environment
export sample_task, get_array_of_tasks

include("meta_learning.jl")
export fomaml
export learning_to_learn_around_a_common_mean, learning_to_learn_around_a_common_mean_r0

include("kernels.jl")
export MaximumMeanDiscrepancy, MMDKernel, kernelmatrix

include("optimisation.jl")
export EmpiricalDistributionKernelFrankWolfe, EmpiricalDistributionKernelHerding, optimize!

include("distributions.jl")
export CategoricalDistribution

end
