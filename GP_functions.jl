using StatsBase
using Random, Distributions
using Optim
using LinearAlgebra
using StatsFuns

function fTraining_mat(N, LB, UB)
    scale = UB .- LB
    rand_mat = rand(D, N)
    training_mat = LB .+ rand_mat .* scale
    return training_mat
end

function compute_kernel(x, x_prime, kernel_para)
    return SE_kernel(x, x_prime, kernel_para)
end

function SE_kernel(x, x_prime, kernel_para)
    s2 = exp(kernel_para[end])
    l = exp.(kernel_para[1:end-1])
    return s2 * exp(-0.5 * sum(((x .- x_prime) ./ l) .^ 2))
end

function RQ_kernel(x, x_prime, kernel_para)
    s2 = exp(kernel_para[end])
    l = exp.(kernel_para[2:end-1])
    α = exp.(kernel_para[1])
    r = sum(((x .- x_prime) ./ 2.0 .* α .* l) .^ 2)
    return s2 * (1.0 + r)^(-α)
end

function NN_kernel(x, x_prime, kernel_para)
    s2 = exp(kernel_para[end])
    Σ = exp.(kernel_para[1:end-1])
    return 2.0 * s2 / π * asin(2.0 * (Σ[1] + sum(x .* Σ[2:end] .* x_prime)) /
                               (sqrt(1.0 + 2.0 * (Σ[1] + sum(x .* Σ[2:end] .* x))) * sqrt(1.0 + 2.0 * (Σ[1] + sum(x_prime .* Σ[2:end] .* x_prime)))))
end

function mNN_kernel(x, x_prime, kernel_para)
    τ = (kernel_para[1:D]) .^ 5.0
    s2 = exp(kernel_para[end])
    Σ = exp.(kernel_para[D+1:end-1])
    x_tilde = (x .- τ)
    x_prime_tilde = (x_prime .- τ)
    return 2.0 * s2 / π * asin(2.0 * (Σ[1] + sum(x_tilde .* Σ[2:end] .* x_prime_tilde)) /
                               (sqrt(1.0 + 2.0 * (Σ[1] + sum(x_tilde .* Σ[2:end] .* x_prime_tilde))) * sqrt(1.0 + 2.0 * (Σ[1] + sum(x_prime_tilde .* Σ[2:end] .* x_prime_tilde)))))
end

function Mat32_kernel(x, x_prime, kernel_para)
    s2 = exp(kernel_para[end])
    l = exp.(kernel_para[1:end-1])
    r = sqrt.(sum((x .- x_prime) ./ l) .^ 2.0)
    return s2 * (1.0 + sqrt(3) .* r) * exp(-sqrt(3) .* r)
end

function Mat52_kernel(x, x_prime, kernel_para)
    s2 = exp(kernel_para[end])
    l = exp.(kernel_para[1:end-1])
    r = sqrt.(sum((x .- x_prime) ./ l) .^ 2.0)
    return s2 * (1.0 + sqrt(5) * sum(sqrt(5) .* sqrt.(r) ./ l) + sqrt(5) * 5.0 / 3.0 * sum(5 / 3 * r ./ l .^ 2.0)) * exp(-sqrt(5) * sum(sqrt.(r) ./ l))
end

function MNN_kernel(x, x_prime, kernel_para)
    s2 = exp(kernel_para[end])
    τ = zeros(D + 1)
    τ[3] = kernel_para[end-1]
    Σ = diagm(exp.(kernel_para[1:D+1]))
    x_tilde = vcat(1.0, x) .- τ
    x_prime_tilde = vcat(1.0, x_prime) .- τ
    return 2.0 * s2 / π * asin(2.0 * x_tilde' * Σ * x_prime_tilde /
                               (sqrt(1.0 + 2.0 * x_tilde' * Σ * x_tilde) * sqrt(1.0 + 2.0 * x_prime_tilde' * Σ * x_prime_tilde)))
end





function generate_K(mx, gp_para)
    kernel_para = gp_para[1:end-1]
    N_x = size(mx)[2]
    K = Matrix{typeof(compute_kernel(mx[:, 1], mx[:, 1], kernel_para))}(undef, N_x, N_x)
    for i = 1:N_x
        for j = 1:i
            k_val = compute_kernel(mx[:, i], mx[:, j], kernel_para)
            K[i, j] = k_val
            K[j, i] = k_val
        end
    end
    se = exp(gp_para[end])
    K = K + se^2.0 * I
    return K
end




function generate_a(mK, vt, vm=0.0)
    # vθ = gp_para[1:D+1]
    try
        mL = cholesky(mK)
        a = mL.U \ (mL.L \ (vt .- vm))
        return a
    catch
        a = mK \ (vt .- vm)
        return a
    end

end

function generate_∂MLover∂θd(mx, vt, gp_para)
    gradient = Array{Float64}(undef, length(gp_para))
    mK_local = generate_K(mx, gp_para)
    mL = cholesky(mK_local)
    a = mL.U \ (mL.L \ (vt))
    α = a * a'
    for i = eachindex(gp_para[1:end-1])
        ∂Kover∂θd = generate_∂Kover∂θd(mx, gp_para, i)
        gradient[i] = 0.5 * tr(α * ∂Kover∂θd - mL.U \ (mL.L \ ∂Kover∂θd))
    end
    ∂Kover∂θd = diagm(ones(length(vt)) .* 2.0 * exp(2.0 * gp_para[end]))
    gradient[end] = 0.5 * tr(α * ∂Kover∂θd - mL.U \ (mL.L \ ∂Kover∂θd))
    return gradient
end



function compute_closeness(l)
    D_l = length(l)
    l = exp.(l)
    l = l ./ sum(l)
    return sum(l .* log.(l ./ (1.0 ./ D_l)))
end

function generate_initial_guess(mx, kernel_LB, kernel_UB, noise_LB, noise_UB)

    D_kernel = length(kernel_LB)
    kernel_init = kernel_LB .+ (kernel_UB .- kernel_LB) .* rand(D_kernel)
    noise_init = noise_LB .+ (noise_UB .- noise_LB) .* rand()
    local θvec_guess = vcat(kernel_init, noise_init)
    K = generate_K(mx, θvec_guess)
    det_K = det(K)
    while det_K == 0.0
        kernel_init = kernel_LB .+ (kernel_UB .- kernel_LB) .* rand(D_kernel)
        noise_init = noise_LB .+ (noise_UB .- noise_LB) .* rand()
        local θvec_guess = vcat(kernel_init, noise_init)
        # println(θvec_guess)
        K = generate_K(mx, θvec_guess)
        det_K = det(K)
    end
    return θvec_guess


end

function fPosterior_mean(x, mx, va, gp_para)
    # se = gp_para[end]
    kernel_para = gp_para[1:end-1]
    result = 0.0
    N_x = size(mx)[2]
    for i = 1:N_x
        result += va[i] * compute_kernel(x, mx[:, i], kernel_para)
    end
    return result
end



function fPosterior_variance(x, mx, mK, gp_para)

    kernel_para = gp_para[1:end-1]
    se = gp_para[end]
    N_x = size(mx)[2]
    kstar = Array{Float64}(undef, N_x)

    for i = 1:N_x
        kstar[i] = compute_kernel(x, mx[:, i], kernel_para)
    end
    return compute_kernel(x, x, kernel_para) - kstar' * (mK \ kstar)
end

function length_function(x_d, c1, c2)
    return exp(c1 * x_d) + exp(c2)
    # return 1.0 / (1.0 + exp(c1 * x_d)) + exp(c2)
end

function Gibbs_kernel(x, x_prime, kernel_para)
    dimension = D
    c1_vec = kernel_para[1:dimension]
    c2_vec = kernel_para[dimension+1:end-1]
    s2 = exp(kernel_para[end])
    result = 1.0
    for i = 1:dimension
        exp_term = sum(
            ((x .- x_prime) .^ 2.0 ./
             (length_function.(x, c1_vec, c2_vec) .^ 2.0 .+
              length_function.(x_prime, c1_vec, c2_vec) .^ 2.0)))
        result = result *
                 (2.0 * length_function(x[i], c1_vec[i], c2_vec[i]) * length_function(x_prime[i], c1_vec[i], c2_vec[i]) /
                  (length_function(x[i], c1_vec[i], c2_vec[i]) + length_function(x_prime[i], c1_vec[i], c2_vec[i]))) *
                 exp(-exp_term)
    end
    return s2 * result
end


function compute_marginal_likelihood(mx, vt, gp_para_log)
    gp_para = gp_para_log
    mK = generate_K(mx, gp_para)


    try
        mK_chol = cholesky(mK)
        α = mK_chol.U \ (mK_chol.L \ vt)
        log_det = sum(log.(diag(mK_chol.U))) * 2
        LL = -0.5 * (vt)' * α - 0.5 * log_det

        return LL
    catch
        mK = mK .+ diagm(ones(size(mK)[1]) .* 1e-6)
        mK_chol = cholesky(mK)
        α = mK_chol.U \ (mK_chol.L \ vt)
        log_det = sum(log.(diag(mK_chol.U))) * 2
        LL = -0.5 * (vt)' * α - 0.5 * log_det

        return LL
    end



end

function compute_approximate_marginal_likelihood(mx, vt, gp_para_log)
    gp_para = gp_para_log
    idx = sample(axes(mx, 2), 10 * D, replace=false)
    vt = vt[idx]
    mx = mx[:, idx]
    mK = generate_K(mx, gp_para)


    try
        mK_chol = cholesky(mK)
        α = mK_chol.U \ (mK_chol.L \ vt)
        log_det = sum(log.(diag(mK_chol.U))) * 2
        LL = -0.5 * (vt)' * α - 0.5 * log_det

        return LL
    catch
        mK = mK .+ diagm(ones(size(mK)[1]) .* 1e-6)
        mK_chol = cholesky(mK)
        α = mK_chol.U \ (mK_chol.L \ vt)
        log_det = sum(log.(diag(mK_chol.U))) * 2
        LL = -0.5 * (vt)' * α - 0.5 * log_det

        return LL
    end



end


function optim_ML(mx, vt, gp_para_log_guess, λ = 0.0,
    kernel_LB=nothing, kernel_UB=nothing,
    noise_LB=nothing, noise_UB=nothing)

    gp_para_LB = vcat(kernel_LB, noise_LB)
    gp_para_UB = vcat(kernel_UB, noise_UB)

    obj(gp_para_log) = -1.0 * compute_marginal_likelihood(mx, vt, gp_para_log) + 
                       0.0 * exp(gp_para_log[end])

    # println(obj(gp_para_log_guess))
    total_iter = 0

    finish = false
    while (~finish) && (total_iter <= 1000)

        try
            rresult = Optim.optimize(obj,
                gp_para_log_guess,
                NelderMead(),
                Optim.Options(show_trace=false, iterations=50))

            gp_para_log_guess = copy(rresult.minimizer)
            finish = rresult.iterations != 50
            total_iter += 50
            # if -1.0 * obj(gp_para_log_guess) < 0.0
            #     return [-1.0 * obj(gp_para_log_guess), gp_para_log_guess]
            # end
        catch
            finish = true
        end
    end

    # if -1.0 * obj(gp_para_log_guess) < LL_0 * 0.95
    #     return [-1.0 * obj(gp_para_log_guess), gp_para_log_guess]
    # end


    # obj_2(gp_para_log) = -1.0 * compute_approximate_marginal_likelihood(mx, vt, gp_para_log) + 
    #     λ * compute_closeness(gp_para_log[1:end-2]) + 
    #     0.0 * exp(gp_para_log[end])

    # # println(obj(gp_para_log_guess))
    # total_iter = 0

    # finish = false
    # while (~finish) && (total_iter <= 300)

    #     try
    #     rresult = Optim.optimize(obj_2, 
    #         gp_para_log_guess, 
    #         NelderMead(), 
    #         Optim.Options(show_trace = false, iterations = 50))

    #     gp_para_log_guess = copy(rresult.minimizer)
    #     finish = rresult.iterations != 50
    #     total_iter += 50
    #     catch
    #         finish = true
    #     end
    # end
    max_LL = -1.0 * obj(gp_para_log_guess)
    return [max_LL, gp_para_log_guess]
end


function optim_ML_2(mx, vt, gp_para_log_guess, λ=0.0,
    kernel_LB=nothing, kernel_UB=nothing,
    noise_LB=nothing, noise_UB=nothing)

    gp_para_LB = vcat(kernel_LB, noise_LB)
    gp_para_UB = vcat(kernel_UB, noise_UB)

    obj(gp_para_log) = -1.0 * compute_marginal_likelihood(mx, vt, gp_para_log)

    function g!(G, gp_para_log)
        G[:] = ForwardDiff.gradient(obj, gp_para_log)
    end

    total_iter = 0

    finish = false
    while (~finish) && (total_iter <= 200000)

        try

            

            rresult = Optim.optimize(obj, g!,
                gp_para_log_guess,
                ConjugateGradient(),
                Optim.Options(show_trace=true, iterations=1000))

            gp_para_log_guess = copy(rresult.minimizer)
            finish = rresult.iterations != 1000
            total_iter += 1000
            if -1.0 * obj(gp_para_log_guess) < 0.0
                return [-1.0 * obj(gp_para_log_guess), gp_para_log_guess]
            end
        catch
            finish = true
        end
    end
    max_LL = -1.0 * obj(gp_para_log_guess)
    return [max_LL, gp_para_log_guess]
end

function optim_ML_3(X, vt, gp_para_log_guess, λ=0.0,
    kernel_LB=nothing, kernel_UB=nothing,
    noise_LB=nothing, noise_UB=nothing)

    buffer = copy(gp_para_log_guess)

    function calculate_common!(x, last_x)
        copy!(last_x, x)
    end

    function obj(para_vec)
        obj_value = -1.0 * comput_marginal_likelihood(X, vt,
            para_vec[1:kernel_length+1])
        # calculate_common!(para_vec, buffer)
        return obj_value
    end
    

    function g!(G, gp_para_log)
        calculate_common!(gp_para_log, buffer)
        G[:] = ForwardDiff.gradient(obj, gp_para_log)
    end

    try
        rresult = Optim.optimize(obj,
            gp_para_log_guess,
            NelderMead(),
            Optim.Options(show_trace=false, iterations=200, g_tol=1e-2, time_limit=30.0))

        gp_para_log_guess = copy(rresult.minimizer)
        if -1.0 * obj(gp_para_log_guess) < 0.0
            return [-1.0 * obj(gp_para_log_guess), gp_para_log_guess]
        elseif (-1.0 * obj(gp_para_log_guess) > 1e6)
            return [0.0, gp_para_log_guess]
        end

        rresult = Optim.optimize(obj, g!,
            gp_para_log_guess,
            ConjugateGradient(),
            Optim.Options(show_trace=false, iterations=5000))
    catch
    end
    gp_para_log_guess = copy(buffer)
    max_LL = -1.0 * obj(gp_para_log_guess)
    return [max_LL, gp_para_log_guess]

end

# function optim_ML(mx, vt, gp_para_log_guess, λ, 
#     kernel_LB=nothing, kernel_UB=nothing,
#     noise_LB=nothing, noise_UB=nothing)

#     gp_para_LB = vcat(kernel_LB, noise_LB)
#     gp_para_UB = vcat(kernel_UB, noise_UB)
#     obj(gp_para_log) = -1.0 * compute_marginal_likelihood(mx, vt, gp_para_log) + 
#         λ * compute_closeness(gp_para_log[1:end-2]) + 
#         0.0 * exp(gp_para_log[end])

#     # println(obj(gp_para_log_guess))
#     total_iter = 0

#     finish = false
#     while (~finish) && (total_iter <= 1000)

#         try
#         rresult = Optim.optimize(obj, 
#             gp_para_log_guess, 
#             NelderMead(), 
#             Optim.Options(show_trace = false, iterations = 50))

#         gp_para_log_guess = copy(rresult.minimizer)
#         finish = rresult.iterations != 50
#         total_iter += 50
#         catch
#             finish = true
#         end
#     end
#     max_LL = -1.0 * obj(gp_para_log_guess)
#     return [max_LL, gp_para_log_guess]
# end

function compute_LOO(mx, vt, gp_para_log)
    gp_para_2 = gp_para_log
    N_x = size(mx)[2]
    mK_2 = generate_K(mx, gp_para_2)
    mL_2 = cholesky(mK_2)
    mKinv = mL_2.U \ (mL_2.L \ I)
    vyinv = mKinv * vt
    result = zeros(length(vt))
    for i = 1:N_x
        g_i = vyinv[i] / mKinv[i, i]
        σsq_i = 1.0 / mKinv[i, i]

        result[i] = -0.5 * log(σsq_i) - (g_i)^2.0 / (2.0 * σsq_i)
    end
    return result

end


function compute_LOO_CV(mx, vt, gp_para_log)
    gp_para_2 = gp_para_log
    N_x = size(mx)[2]
    mK_2 = generate_K(mx, gp_para_2)
    # try
    #     mL_2 = cholesky(mK_2)
    #     mKinv = mL_2.U \ (mL_2.L \ I)  
    # catch
    #     mKinv = pinv(mK_2)
    # end
    mKinv = pinv(mK_2)
    vyinv = mKinv * vt
    result = 0.0
    for i = 1:N_x
        g_i = vyinv[i] / mKinv[i, i]
        σsq_i = 1.0 / mKinv[i, i]
        result += -0.5 * log(σsq_i) - (g_i)^2.0 / (2.0 * σsq_i)
    end
    return result

end

function optim_LOO_CV(mx, vt, gp_para_log_guess,
    kernel_LB=nothing, kernel_UB=nothing,
    noise_LB=nothing, noise_UB=nothing)

    gp_para_LB = vcat(kernel_LB, noise_LB)
    gp_para_UB = vcat(kernel_UB, noise_UB)
    obj(gp_para_2) = -1.0 * compute_LOO_CV(mx, vt, gp_para_2)
    rresult = Optim.optimize(obj, gp_para_LB, gp_para_UB,
        gp_para_log_guess,
        NelderMead(),
        Optim.Options(show_trace=false, g_tol=5e-3))
    gp_para = rresult.minimizer
    max_LL = -1.0 * rresult.minimum
    return [max_LL, gp_para]
end

# function optim_LOO_SE(mx, vt, gp_para_log_guess, q, 
#     kernel_LB=nothing, kernel_UB=nothing,
#     noise_LB=nothing, noise_UB=nothing)

#     gp_para_LB = vcat(kernel_LB, noise_LB)
#     gp_para_UB = vcat(kernel_UB, noise_UB)
#     N =  trunc(Int, length(vt) * q)
#     obj(gp_para_2) = -1.0 * sum(sort(compute_LOO(mx, vt, gp_para_2))[1:N])
#     rresult = Optim.optimize(obj, 
#         gp_para_log_guess, 
#         NelderMead(), 
#         Optim.Options(show_trace = false, g_tol = 5e-3))
#     gp_para = rresult.minimizer
#     max_LL = -1.0 * rresult.minimum
#     return [max_LL, gp_para]
# end



function logit_function(x, lb, ub)
    x_normal = (x - lb) / (ub - lb)
    return log(x_normal / (1.0 - x_normal))
end

function inverse_logit_function(y, lb, ub)
    x1 = 1.0 / (1.0 + exp(-y))
    return x1 * (ub - lb) + lb
end

function bound_normalize(X, LB, UB)
    return (X .- LB) ./ (UB .- LB) .* 2.0 .- 1.0
end

function bound_normalize_2(X, LB, UB)
    return (X .- LB) ./ (UB .- LB) .+ 1.0
end


function std_normalize(X, mean, std)
    return (X .- mean) ./ std
end

function inverse_bound_normalize(y, LB, UB)
    return (y .+ 1.0) ./ 2.0 .* (UB .- LB) .+ LB
end

function inverse_bound_normalize_2(y, LB, UB)
    return (y .- 1.0) .* (UB .- LB) .+ LB
end


function inverse_std_normalize(y, mean, std)
    return y .* std .+ mean
end



function bound_normalize_mat(X_mat, LB, UB)
    X_normalized = Array{Float64}(undef, size(X_mat))
    N_x = size(X_mat)[2]
    for i = 1:N_x
        X_normalized[:, i] = bound_normalize(X_mat[:, i], LB, UB)
    end
    return X_normalized
end

function std_normalize_mat(X_mat, mean, std)
    X_normalized = Array{Float64}(undef, size(X_mat))
    N_x = size(X_mat)[2]
    for i = 1:N_x
        X_normalized[:, i] = bound_normalize(X_mat[:, i], mean, std)
    end
    return X_normalized
end

function optim_GP(training_inputs, training_outputs, θvec_guess,
    kernel_LB=nothing, kernel_UB=nothing,
    noise_LB=nothing, noise_UB=nothing)

    kernel_init = copy(θvec_guess[1:end-1])
    noise_init = copy(θvec_guess[end])
    kernel2 = kernel_function(kernel_init[1:end-1], kernel_init[end])
    logObsNoise2 = noise_init

    gp2 = GP(training_inputs, training_outputs, mzero, kernel2, logObsNoise2)
    Optim.optimize!(gp2, noisebounds=[noise_LB, noise_UB], Kernbounds=[kernel_LB, kernel_UB]; autodiff=:forward)


    LL = gp2.mll

    temp_kernel = gp2.kernel
    temp_logNoise = gp2.logNoise
    θvec_final = vcat(log.(temp_kernel.iℓ2) ./ -2,
        log(7.38905609893065, temp_kernel.σ2), temp_logNoise.value)

    return [LL, θvec_final]
end


function gp_predict(x, mx, va, gp_para, LB, UB)
    x_normarlized = bound_normalize(x, LB, UB)
    y_predict = fPosterior_mean(x_normarlized, mx, va, gp_para)
    # return inverse_logit_function(inverse_std_normalize(y_predict, mean, std), lb, ub)
    return y_predict
end


function gp_predict(x, mx, va, gp_para, mean, std, LB, UB)
    x_normarlized = bound_normalize(x, LB, UB)
    y_predict = fPosterior_mean(x_normarlized, mx, va, gp_para)
    # return inverse_logit_function(inverse_std_normalize(y_predict, mean, std), lb, ub)
    return inverse_std_normalize(y_predict, mean, std)
end

function gp_predict_2(x, mx, va, gp_para, mean, std, LB, UB)
    x_normarlized = bound_normalize(x, LB, UB)
    y_predict = fPosterior_mean(x_normarlized, mx, va, gp_para)
    y_var = fPosterior_variance(x_normarlized, mx, mK, gp_para)
    # return inverse_logit_function(inverse_std_normalize(y_predict, mean, std), lb, ub)
    return inverse_std_normalize(y_predict, mean, std) - 1000.0 * y_var
end


function gp_predict(x, mx, va, gp_para, mean, std, LB, UB, lb, ub)
    x_normarlized = bound_normalize(x, LB, UB)
    y_predict = exp(fPosterior_mean(x_normarlized, mx, va, gp_para))
    # return inverse_logit_function(inverse_bound_normalize_2(y_predict, mean, std), lb, ub)
    return inverse_bound_normalize_2(y_predict, lb, ub)
end

function gp_predict(x, mx, va, gp_para, LB, UB)
    x_normarlized = bound_normalize(x, LB, UB)
    y_predict = fPosterior_mean(x_normarlized, mx, va, gp_para)
    return y_predict
end

# function gp_predict_2(x, mx, va, gp_para, mean, std,)
#     y_predict = fPosterior_mean(x, mx, va, gp_para)
#     # return inverse_logit_function(inverse_std_normalize(y_predict, mean, std), lb, ub)
#     return inverse_std_normalize(y_predict, mean, std)
# end

function gp_predict_2(x, mx, va, gp_para, mean, std, LB, UB, lb=nothing, ub=nothing)
    x_normarlized = bound_normalize(x, LB, UB)
    x_normarlized[6] = logit_function(x, -1.0, 1.0)
    y_predict = fPosterior_mean(x_normarlized, mx, va, gp_para)
    # return inverse_logit_function(inverse_std_normalize(y_predict, mean, std), lb, ub)
    return inverse_std_normalize(y_predict, mean, std)
end


function train_gp(training_inputs, training_outputs, initial_guess, Likelihood_iter, λ=0.0)
    found_vec = SharedArray{Int64}(1)

    θvec_local = SharedArray{Float64}(size(initial_guess))
    θvec_local[:] = initial_guess

    MaxLikelihood = SharedArray{Float64}(1)
    MaxLikelihood[1] = 0.0

    while found_vec[1] < Likelihood_iter


        @sync @distributed for i = 1:Likelihood_iter
            if i == 1
                θvec_guess = copy(θvec_local) + rand(length(θvec_local)) * 1e-2
                # θvec_guess = copy(θvec_local)
            else
                kernel_init = kernel_LB .+ (kernel_UB .- kernel_LB) .* rand(length(kernel_LB))
                noise_init = noise_LB .+ (noise_UB .- noise_LB) .* rand()
                θvec_guess = vcat(kernel_init, noise_init)
            end


            try
                LLresult = optim_ML(training_inputs,
                    training_outputs, θvec_guess, λ,
                    kernel_LB, kernel_UB,
                    noise_LB, noise_UB)

                # if LLresult[1] > 1e5
                #     continue
                # end
                if LLresult[1] > MaxLikelihood[1]

                    θvec_local[1:end] = copy(LLresult[2])

                    MaxLikelihood[1] = LLresult[1]
                    # println(LLresult[1])
                    # println("Found sth here")

                else
                    # println(LLresult[1])
                    # println("Found Nothing here")
                end
                found_vec[1] += 1
            catch
                println("GP optim error")
                continue
            end

        end

    end
    println(MaxLikelihood[1])
    return [MaxLikelihood[1], θvec_local]

end


function train_gp(training_inputs, training_outputs, initial_guesses, Likelihood_iter)
    found_vec = SharedArray{Int64}(1)

    θvec_local = SharedArray{Float64}(size(initial_guesses[:, 1]))
    θvec_local[:] = initial_guesses[:, 1]

    MaxLikelihood = SharedArray{Float64}(1)
    MaxLikelihood[1] = 0.0

    while found_vec[1] < Likelihood_iter


        @sync @distributed for i = 1:Likelihood_iter
            θvec_guess = initial_guesses[:, i]
            try
                LLresult = optim_ML(training_inputs,
                    training_outputs, θvec_guess, 0.0,
                    kernel_LB, kernel_UB,
                    noise_LB, noise_UB)

                # if LLresult[1] > 1e5
                #     continue
                # end
                if LLresult[1] > MaxLikelihood[1]

                    θvec_local[1:end] = copy(LLresult[2])

                    MaxLikelihood[1] = LLresult[1]
                    # println(LLresult[1])
                    # println("Found sth here")

                else
                    # println(LLresult[1])
                    # println("Found Nothing here")
                end
                found_vec[1] += 1
            catch
                println("GP optim error")
                continue
            end

        end

    end
    println(MaxLikelihood[1])
    return [MaxLikelihood[1], θvec_local]

end


function train_gp_LOO(training_inputs, training_outputs, initial_guess, Likelihood_iter, q, λ=0.0)
    found_vec = SharedArray{Int64}(1)

    θvec_local = SharedArray{Float64}(size(initial_guess))
    θvec_local[:] = initial_guess

    MaxLikelihood = SharedArray{Float64}(1)
    MaxLikelihood[1] = 0.0

    while found_vec[1] < Likelihood_iter


        @sync @distributed for i = 1:Likelihood_iter
            if i == 1
                θvec_guess = copy(θvec_local) + rand(length(θvec_local)) * 1e-2
                # θvec_guess = copy(θvec_local)
            else
                kernel_init = kernel_LB .+ (kernel_UB .- kernel_LB) .* rand(length(kernel_LB))
                noise_init = noise_LB .+ (noise_UB .- noise_LB) .* rand()
                θvec_guess = vcat(kernel_init, noise_init)
            end


            try
                LLresult = optim_LOO_CV(training_inputs,
                    training_outputs, θvec_guess, q)

                # if LLresult[1] > 1e5
                #     continue
                # end
                if LLresult[1] > MaxLikelihood[1]

                    θvec_local[1:end] = copy(LLresult[2])

                    MaxLikelihood[1] = LLresult[1]
                    # println(LLresult[1])
                    # println("Found sth here")

                else
                    # println(LLresult[1])
                    # println("Found Nothing here")
                end
                found_vec[1] += 1
            catch
                println("GP optim error")
                continue
            end

        end

    end
    println(MaxLikelihood[1])
    return [MaxLikelihood[1], θvec_local]

end

function train_gp(training_inputs, training_outputs, initial_guess, Likelihood_iter, λ=0.0, test=1000.0)



    found_vec = SharedArray{Int64}(1)

    θvec_local = SharedArray{Float64}(size(initial_guess))
    θvec_local[:] = initial_guess

    MaxLikelihood = SharedArray{Float64}(1)
    MaxLikelihood[1] = 0.0

    if test > 0.0
        try
            LL_0 = compute_marginal_likelihood(training_inputs, training_outputs, initial_guess)
            if LL_0 > test * 0.99 &&  (LL_0 > 100.0)
                return [LL_0, θvec_local]
            end
        catch
        end
    end

    while found_vec[1] < Likelihood_iter


        @sync @distributed for i = 1:Likelihood_iter
            if i == 1
                θvec_guess = copy(θvec_local) + rand(length(θvec_local)) * 1e-2
                # θvec_guess = copy(θvec_local)
            else
                kernel_init = kernel_LB .+ (kernel_UB .- kernel_LB) .* rand(length(kernel_LB))
                noise_init = noise_LB .+ (noise_UB .- noise_LB) .* rand()
                θvec_guess = vcat(kernel_init, noise_init)
            end


            # try
                LLresult = optim_ML(training_inputs,
                    training_outputs, θvec_guess, λ,
                    kernel_LB, kernel_UB,
                    noise_LB, noise_UB)

                # if LLresult[1] > 1e5
                #     continue
                # end
                if LLresult[1] > MaxLikelihood[1]

                    θvec_local[1:end] = copy(LLresult[2])

                    MaxLikelihood[1] = LLresult[1]
                    # println(LLresult[1])
                    # println("Found sth here")

                else
                    # println(LLresult[1])
                    # println("Found Nothing here")
                end
                found_vec[1] += 1
            # catch
            #     println("GP optim error")
            #     continue
            # end

        end

    end
    println(MaxLikelihood[1])
    return [MaxLikelihood[1], θvec_local]

end


# function train_gp(training_inputs, training_outputs, test_inputs, test_outputs, initial_guess, Likelihood_iter, λ = 0.0)
#     found_vec = SharedArray{Int64}(1)

#     θvec_local = SharedArray{Float64}(size(initial_guess))
#     θvec_local[:] = initial_guess

#     MaxLikelihood = SharedArray{Float64}(1)
#     MaxLikelihood[1] = 0.0

#     while found_vec[1] < Likelihood_iter


#         @sync @distributed for i = 1:Likelihood_iter
#             if i == 1
#                 θvec_guess = copy(θvec_local) + rand(length(θvec_local)) * 1e-2
#                 # θvec_guess = copy(θvec_local)
#             else
#                 kernel_init = kernel_LB .+ (kernel_UB .- kernel_LB) .* rand(length(kernel_LB))
#                 noise_init = noise_LB .+ (noise_UB .- noise_LB) .* rand()
#                 θvec_guess = vcat(kernel_init, noise_init)
#             end

#             try
#                 LLresult = optim_ML(training_inputs, 
#                     training_outputs, θvec_guess, λ,
#                     kernel_LB, kernel_UB,
#                     noise_LB, noise_UB)

#                 # if LLresult[1] > 1e5
#                 #     continue
#                 # end

#                 test_LL = compute_marginal_likelihood(test_inputs, test_outputs, LLresult[2])
#                 if test_LL > MaxLikelihood[1]

#                     θvec_local[1:end] = copy(LLresult[2])

#                     MaxLikelihood[1] = test_LL
#                     # println(test_LL)
#                     # println("Found sth here")

#                 else
#                     # println(test_LL)
#                     # println("Found Nothing here")
#                 end
#                 found_vec[1] += 1
#             catch
#                 println("GP optim error")
#                 continue
#             end

#         end

#     end
#     println(MaxLikelihood[1])
#     return [MaxLikelihood[1], θvec_local]

# end

function select_λ(inputs, outputs, initial_guess)
    λ_list = [0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    λ_result_list = Array{Float64}(undef, length(λ_list))
    for i = 1:length(λ_list)
        LL = 0.0
        λ = λ_list[i]
        for j = 1:5
            sample = rand(1:length(outputs), trunc(Int, length(outputs) * 0.5))
            θvec = train_gp(inputs[:, sample], outputs[sample], initial_guess, 5, λ)[2]
            LL += compute_marginal_likelihood(inputs, outputs, θvec)
        end
        λ_result_list = LL

    end
    return λ_list[findmax(λ_result_list)[2]]
end

function select_λ(training_inputs, training_outputs, inputs, outputs, initial_guess)
    λ_list = [0.0, 1.0, 10.0, 100.0]
    λ_result_list = Array{Float64}(undef, length(λ_list))
    for i = 1:length(λ_list)
        λ = λ_list[i]
        θvec = train_gp(training_inputs, training_outputs, initial_guess, 5)[2]
        λ_result_list[i] = compute_marginal_likelihood(inputs, outputs, θvec)

    end
    return λ_list[findmax(λ_result_list)[2]]

end


function fTraining_mat(N, LB, UB)
    scale = UB .- LB
    rand_mat = rand(D, N)
    training_mat = LB .+ rand_mat .* scale
    return training_mat
end

# function fTraining_normal_mat(N, LB, UB)
#     scale = UB .- LB
#     z_entropy = 0.5 * log(2 * π * 1.0^2.0) + 0.5
#     sample_entropy = 0.0
#     while abs(sample_entropy - z_entropy) >= 0.0001
#         global z_draws = rand(Normal(), N * D)
#         sample_entropy = 0.0
#         sample_entropy =
#             -sum(log.(normpdf.(z_draws)) ./ (N * D))
#     end
#     rand_mat = reshape(z_draws, D, N)
#     rand_mat_max = maximum(rand_mat) * 1.05
#     rand_mat_min = minimum(rand_mat) * 1.05
#     rand_mat = (rand_mat .- rand_mat_min) ./ (rand_mat_max - rand_mat_min)
#     training_mat = LB .+ rand_mat .* scale
#     return training_mat
# end


function fTraining_normal_mat(N, LB, UB)
    scale = UB .- LB

    sample_entropy = 10.0
    means = zeros(D)
    stds = I(D)
    normal_dist = MvNormal(means, stds)
    z_entropy = entropy(normal_dist)
    local rand_mat = Array{Float64}(undef, D, N)
    while (abs(sample_entropy - z_entropy) >= 0.000001)
        rand_mat = rand(normal_dist, N)
        sample_entropy = 0.0
        for i = 1:N
            sample_entropy -= logpdf(normal_dist, rand_mat[:, i])
        end
        sample_entropy = sample_entropy / N



    end
    # println("typ set")
    # for i = 1:N_inputs
    #     local knn_dist_i = compute_knn_distance(rand_mat[:, i], rand_mat)
    #     # println(i, " ", knn_dist_i)
    #     while  knn_dist_i > 2.0
    #         rand_mat[:, i] = rand(normal_dist, 1)
    #         knn_dist_i = compute_knn_distance(rand_mat[:, i], rand_mat)
    #     end   

    # end

    rand_mat_max = [minimum(skipmissing(rand_mat[row, :])) for row in 1:size(rand_mat)[1]] .* 1.05
    rand_mat_min = [maximum(skipmissing(rand_mat[row, :])) for row in 1:size(rand_mat)[1]] .* 1.05
    rand_mat = (rand_mat .- rand_mat_min) ./ (rand_mat_max - rand_mat_min)


    # rand_mat = (rand_mat .- LB) ./ (UB - LB)
    training_mat = LB .+ rand_mat .* scale
    return training_mat
end

function find_typical_set(N)
    sample_entropy = 10.0
    means = zeros(D)
    stds = I(D)
    normal_dist = MvNormal(means, stds)
    z_entropy = entropy(normal_dist)
    local rand_mat = Array{Float64}(undef, D, N)
    while (abs(sample_entropy - z_entropy) >= 0.000001)
        rand_mat = rand(normal_dist, N)
        sample_entropy = 0.0
        for i = 1:N
            sample_entropy -= logpdf(normal_dist, rand_mat[:, i])
        end
        sample_entropy = sample_entropy / N

    end
    return rand_mat
end

function find_cluster_mat(N, LB, UB)
    scale = UB .- LB
    centres = find_typical_set(N ÷ 20)
    local rand_mat = Array{Float64}(undef, D, N)
    for i = 1:(N÷20)
        cluster = centres[:, i] .+ find_typical_set(20) ./ 3.0
        rand_mat[:, 1+(i-1)*20:i*20] = cluster
    end

    rand_mat_max = [minimum(skipmissing(rand_mat[row, :])) for row in 1:size(rand_mat)[1]] .* 1.05
    rand_mat_min = [maximum(skipmissing(rand_mat[row, :])) for row in 1:size(rand_mat)[1]] .* 1.05
    rand_mat = (rand_mat .- rand_mat_min) ./ (rand_mat_max - rand_mat_min)
    rand_mat = rand_mat[:, shuffle(1:N)]
    # rand_mat = (rand_mat .- LB) ./ (UB - LB)
    training_mat = LB .+ rand_mat .* scale
    return training_mat
end


function fKernel_mat(N, kernel_LB, kernel_UB)
    scale = kernel_UB .- kernel_LB
    rand_mat = rand(length(kernel_LB), N)
    training_mat = kernel_LB .+ rand_mat .* scale
    return training_mat
end


function train_gp(training_inputs, training_outputs, initial_guess, Likelihood_iter, λ=0.0, test=0.0)



    found_vec = SharedArray{Int64}(1)

    θvec_local = SharedArray{Float64}(size(initial_guess))
    θvec_local[:] = initial_guess

    MaxLikelihood = SharedArray{Float64}(1)
    MaxLikelihood[1] = 0.0

    if test > 0.0
        try
            LL_0 = compute_marginal_likelihood(training_inputs, training_outputs, initial_guess)
            if (LL_0 > test * 0.99) &&  (LL_0 > 100.0)
                return [LL_0, θvec_local]
            end
        catch
        end
    end

    while found_vec[1] < Likelihood_iter


        @sync @distributed for i = 1:Likelihood_iter
            if i == 1
                θvec_guess = copy(θvec_local) + rand(length(θvec_local)) * 1e-2
                # θvec_guess = copy(θvec_local)
            else
                kernel_init = kernel_LB .+ (kernel_UB .- kernel_LB) .* rand(length(kernel_LB))
                noise_init = noise_LB .+ (noise_UB .- noise_LB) .* rand()
                θvec_guess = vcat(kernel_init, noise_init)
            end


            try
                LLresult = optim_ML(training_inputs,
                    training_outputs, θvec_guess, λ,
                    kernel_LB, kernel_UB,
                    noise_LB, noise_UB)

                # if LLresult[1] > 1e5
                #     continue
                # end
                if compute_marginal_likelihood(training_inputs, training_outputs, LLresult[2]) > MaxLikelihood[1]

                    θvec_local[1:end] = copy(LLresult[2])

                    MaxLikelihood[1] = LLresult[1]
                    # println(LLresult[1])
                    # println("Found sth here")

                else
                    # println(LLresult[1])
                    # println("Found Nothing here")
                end
                found_vec[1] += 1
            catch
                println("GP optim error")
                continue
            end

        end

    end
    println(MaxLikelihood[1])
    return [MaxLikelihood[1], θvec_local]

end