using Distributed
const ncores = 5
addprocs(ncores - 1)

@everywhere using Random
@everywhere using NLopt
@everywhere using JLD2
@everywhere using FileIO
@everywhere using SharedArrays
@everywhere using StatsBase
@everywhere using ForwardDiff
@everywhere include("GP_functions.jl")
@everywhere using Dates
@everywhere dt = Dates.now()

Random.seed!(1234)

@everywhere include("Baseline_Model_Parameters.jl")
@everywhere include("Baseline_Model_Equations.jl")
@everywhere include("Baseline_Model_Simulation.jl")

@everywhere const kernel_length = D + 1
@everywhere const kernel_LB = -10.0 * ones(kernel_length)
@everywhere const kernel_UB = 10.0 * ones(kernel_length)
@everywhere const noise_LB = -12.0
@everywhere const noise_UB = -7.0

@everywhere const dimension_input = 20
@everywhere const optim_size = dimension_input * D
@everywhere const N_test_inputs = 50
@everywhere const N_training_inputs = dimension_input * D
@everywhere const N_inputs = N_training_inputs + N_test_inputs

@everywhere const max_iter = 1000
@everywhere const tol = 1e-5


@everywhere global s_initial = [θ_y_ss, 0.454 + 0.315 / 3.667, 1.38]
@everywhere global s_inputs = fTraining_mat(N_inputs, LB, UB)
@everywhere global s_inputs[:, dimension_input] = s_initial

save("temp\\s_inputs.jld", "data", s_inputs)
@everywhere global s_inputs = load("temp\\s_inputs.jld")["data"]
@everywhere global s_inputs_normarlized = bound_normalize_mat(s_inputs, LB, UB)


@everywhere const control_LBs = [e_LB θ_y_LB k_LB -0.0 -0.0 -0.0;
    e_LB θ_y_LB k_LB -0.0 -0.0 -1.0;
    e_LB θ_y_LB k_LB -1.0 -0.0 -1.0;
    e_LB θ_y_LB k_LB 0.0 -0.0 0.0]'

@everywhere const control_UBs = [e_UB θ_y_UB k_UB -0.0 -0.0 -0.0;
    e_UB θ_y_UB k_UB -0.0 0.0 -0.0;
    e_UB θ_y_UB k_UB -0.0 0.0 -0.0;
    e_UB θ_y_UB k_UB -0.0 10.0 -0.0]'

@everywhere const regimes = ["BAU", "greensubsidyonly", "nocarbontax", "carbontaxonly"]

@everywhere fV_prior(X) = -0.0 # Our trival prior guess; notice X is a vector here
@everywhere va = 0.0
@everywhere gp_para = 0.0

global t_set = SharedArray{Float64}(N_inputs)
global control_set = SharedArray{Float64}(size(control_LBs)[1], N_inputs)
@everywhere global var_set = SharedArray{Float64}(N_inputs)
global c_set = SharedArray{Float64}(N_inputs)
global s_tomorrow_set = SharedArray{Float64}(D, N_inputs)
global ret_set = SharedArray{Float64}(N_inputs)
global loss_1_1_set = SharedArray{Float64}(N_inputs)
global loss_1_2_set = SharedArray{Float64}(N_inputs)
global loss_2_1_set = SharedArray{Float64}(N_inputs)
global loss_2_2_set = SharedArray{Float64}(N_inputs)
@everywhere global LOO = Array{Float64}(undef, N_training_inputs)

# global s_inputs_history = Array{Float64}(undef, D, N_inputs, max_iter)
@everywhere global t_set_history = Array{Float64}(undef, N_inputs, max_iter)
global s_tomorrow_history = Array{Float64}(undef, D, N_inputs, max_iter)
@everywhere global control_set_history = Array{Float64}(undef, size(control_LBs)[1], N_inputs, max_iter)
global loss_1_1_set_history = Array{Float64}(undef, N_inputs, max_iter)
global loss_1_2_set_history = Array{Float64}(undef, N_inputs, max_iter)
global loss_2_1_set_history = Array{Float64}(undef, N_inputs, max_iter)
global loss_2_2_set_history = Array{Float64}(undef, N_inputs, max_iter)
global ret_set_history = Array{Float64}(undef, N_inputs, max_iter)
@everywhere global var_set_history = Array{Float64}(undef, N_inputs, max_iter)

@everywhere global θvec_history = Array{Float64}(undef, kernel_length + 1, max_iter)
@everywhere global θvec_e_history = Array{Float64}(undef, kernel_length + 1, max_iter)
@everywhere global θvec_θ_history = Array{Float64}(undef, kernel_length + 1, max_iter)
@everywhere global θvec_τ_θ_history = Array{Float64}(undef, kernel_length + 1, max_iter)
@everywhere global θvec_τ_B_history = Array{Float64}(undef, kernel_length + 1, max_iter)
@everywhere global θvec_τ_G_history = Array{Float64}(undef, kernel_length + 1, max_iter)
@everywhere global θvec_c_history = Array{Float64}(undef, kernel_length + 1, max_iter)
@everywhere const control_initial_guess = [log(2.3), θ_y_ss, k_LB, 0.0, 0.0, 0.0]
@everywhere global prediction_sample = 1:optim_size

global Results = []
@everywhere global policy_iter = 0


@everywhere global τ_θ_function(s) = 0.0
@everywhere global τ_B_function(s) = 0.0
@everywhere global τ_G_function(s) = 0.0



@everywhere global t = max_iter


while policy_iter < 4
    @everywhere policy_iter = policy_iter + 1
    @everywhere control_LB = copy(control_LBs[:, policy_iter])
    @everywhere control_UB = copy(control_UBs[:, policy_iter])
    @everywhere global t = max_iter
    print_ddist = 10.0
    LL = 0.0
    LL_e = 0.0
    LL_θ = 0.0
    LL_c = 0.0

    @everywhere global fValue_function(s) = fV_prior(s)
    @everywhere global e_function(s) = 1.0
    @everywhere global θ_function(s) = s[1]
    @everywhere global k_function(s) = s[3]
    @everywhere global c_function(s) = 1.0



    @everywhere prediction_sample = prediction_sample
    while t >= 1
        println(t)

        time_start = time()
        # s_inputs_history[:, :, t] = s_inputs
        # t_set_hybrid_raw = Array{Float64}(undef, optim_size)

        @sync @distributed for i = 1:N_inputs
            s_today = s_inputs[:, i]
            if t > max_iter - 1
                control_initial = copy(control_initial_guess)

                control_initial[2] = s_today[1]
                vfi_result = value_function_iteration_no_constraint(
                    control_initial, s_today)

                convergence = 1.0

            else
                control_init = control_set[:, i]
                convergence = 0.0
                vfi_result = value_function_iteration(control_init, s_today)
                convergence = (vfi_result[3] == "XTOL_REACHED") &&
                              (abs(y_tech_forward_constraint(vfi_result[2], s_today)) < 1e-4) &&
                              (abs(capital_euler_constraint(vfi_result[2], s_today)) < 1e-4)
                # println([i, control_init, vfi_result[3], vfi_result[2]])
            end

            if convergence != 1
                # println(i)
                control_init = value_function_iteration_no_constraint(
                    control_initial_guess, s_today)[2]
                vfi_result = value_function_iteration(control_init, s_today)
                convergence = (vfi_result[3] == "XTOL_REACHED")

            end

            control_set[:, i] = copy(vfi_result[2])
            t_set[i] = -1.0 * copy(vfi_result[1])

            c_set[i] = total_consumption(control_set[:, i], s_today)

            # if c_set[i] < 0
            #     global policy_iter = 100.0
            #     @everywhere global t = 0
            # end 
            s_tomorrow = law_of_motion(control_set[:, i], s_today)
            s_tomorrow_set[:, i] = s_tomorrow
            ret_set[i] = convergence

            if convergence != 1
                println(i)

                # control_set[:, i] = copy(control_set_history[:, i, t+1])
                # t_set[i] = copy(fValue_today(control_set_history[:, i, t+1], s_today))
            end


        end



        if t < max_iter
            max_tol = 1e-6
            for i = 1:N_inputs
                s_t = s_inputs[:, i]
                control_init = control_set[:, i]
                vfi_result = value_function_iteration(control_init, s_t)
                convergence_1 = (vfi_result[3] == "XTOL_REACHED")
                x_t = vfi_result[2]
                s_t_2 = law_of_motion(x_t, s_t)
                vfi_result = value_function_iteration(x_t, s_t_2)
                x_t_2 = vfi_result[2]
                convergence_2 = (vfi_result[3] == "XTOL_REACHED")

                try
                    loss_1_1_set[i] = y_tech_forward_constraint(x_t, s_t) * convergence_1 * convergence_2
                    loss_1_2_set[i] = capital_euler_constraint(x_t, s_t) * convergence_1 * convergence_2
                    loss_2_1_set[i] = y_tech_forward_constraint_test(x_t, s_t, x_t_2) * convergence_1 * convergence_2
                    loss_2_2_set[i] = capital_euler_constraint_test(x_t, s_t, x_t_2) * convergence_1 * convergence_2
                catch
                    ret_set[i] = 0
                    println("constraint error")
                    println(i)
                end

            end

            loss_1_1_set_history[:, t] = copy(loss_1_1_set)
            loss_1_2_set_history[:, t] = copy(loss_1_2_set)
            loss_2_1_set_history[:, t] = copy(loss_2_1_set)
            loss_2_2_set_history[:, t] = copy(loss_2_2_set)
        end

        # if minimum(ret_set) < 1.0
        #     break
        # end

        if t < max_iter
            for i = 1:N_inputs
                var_set[i] = fPosterior_variance(bound_normalize(s_tomorrow_set[:, i], LB, UB), s_inputs_normarlized[:, prediction_sample], mK, θvec)
            end


            @everywhere prediction_sample = findall((var_set[1:optim_size] .< 1e-6) .& (ret_set[1:optim_size] .== 1) .& (abs.(c_set[1:optim_size]) .< Inf))
            println(length(prediction_sample))
        end
        # if maximum(ret_set) < 1
        #     break
        # end

        @everywhere t_set_normalized = copy(t_set)
        @everywhere t_mean = mean(t_set_normalized[prediction_sample])
        @everywhere t_std = std(t_set_normalized[prediction_sample])
        @everywhere t_set_normalized = std_normalize.(t_set_normalized, t_mean, t_std)

        @everywhere e_set_normalized = copy(control_set[1, :])
        @everywhere e_mean = mean(e_set_normalized[prediction_sample])
        @everywhere e_std = std(e_set_normalized[prediction_sample])
        @everywhere e_set_normalized = std_normalize.(e_set_normalized, e_mean, e_std)


        @everywhere c_set_normalized = copy(c_set)
        @everywhere c_mean = mean(c_set_normalized[prediction_sample])
        @everywhere c_std = std(c_set_normalized[prediction_sample])
        @everywhere c_set_normalized = std_normalize.(c_set_normalized, c_mean, c_std)


        @everywhere θ_set_normalized = copy(control_set[2, :])
        @everywhere θ_mean = mean(θ_set_normalized[prediction_sample])
        @everywhere θ_std = std(θ_set_normalized[prediction_sample])
        @everywhere θ_set_normalized = std_normalize.(θ_set_normalized, θ_mean, θ_std)

        @everywhere k_set_normalized = copy(control_set[3, :])
        @everywhere k_mean = mean(k_set_normalized[prediction_sample])
        @everywhere k_std = std(k_set_normalized[prediction_sample])
        @everywhere k_set_normalized = std_normalize.(k_set_normalized, k_mean, k_std)


        @everywhere τ_θ_set_normalized = copy(control_set[4, :])
        @everywhere τ_θ_mean = mean(τ_θ_set_normalized[prediction_sample])
        @everywhere τ_θ_std = std(τ_θ_set_normalized[prediction_sample])
        @everywhere τ_θ_set_normalized = std_normalize.(τ_θ_set_normalized, τ_θ_mean, τ_θ_std)

        @everywhere τ_B_set_normalized = copy(control_set[5, :])
        @everywhere τ_B_mean = mean(τ_B_set_normalized[prediction_sample])
        @everywhere τ_B_std = std(τ_B_set_normalized[prediction_sample])
        @everywhere τ_B_set_normalized = std_normalize.(τ_B_set_normalized, τ_B_mean, τ_B_std)

        @everywhere τ_G_set_normalized = copy(control_set[6, :])
        @everywhere τ_G_mean = mean(τ_G_set_normalized[prediction_sample])
        @everywhere τ_G_std = std(τ_G_set_normalized[prediction_sample])
        @everywhere τ_G_set_normalized = std_normalize.(τ_G_set_normalized, τ_G_mean, τ_G_std)


        if t < max_iter - 1

            print_ddist = maximum(abs.(t_set[findall(ret_set .== 1.0)] .- t_set_history[findall(ret_set .== 1.0), t+1]))
            println(print_ddist)
        end


        if t == max_iter
            global θvec_guess = vcat(0.1 * ones(kernel_length), -8.0)
        else
            global θvec_guess = copy(θvec_history[:, t+1])
        end


        if t == max_iter
            train_gp_result = train_gp(s_inputs_normarlized[:, prediction_sample],
                t_set_normalized[prediction_sample], θvec_guess, 30, 0.0, LL)
            θvec_temp = copy(train_gp_result[2])
            LL = train_gp_result[1]

            train_gp_result_e = train_gp(s_inputs_normarlized[:, prediction_sample],
                e_set_normalized[prediction_sample], θvec_temp, 30, 0.0, LL_e)
            θvec_temp_e = copy(train_gp_result_e[2])
            LL_e = train_gp_result_e[1]

            train_gp_result_θ = train_gp(s_inputs_normarlized[:, prediction_sample],
                θ_set_normalized[prediction_sample], θvec_temp, 30, 0.0, LL_θ)
            θvec_temp_θ = copy(train_gp_result_θ[2])
            LL_θ = train_gp_result_θ[1]

            train_gp_result_c = train_gp(s_inputs_normarlized[:, prediction_sample],
                c_set_normalized[prediction_sample], θvec_temp, 30, 0.0, LL_c)
            θvec_temp_c = copy(train_gp_result_c[2])
            LL_c = train_gp_result_c[1]

            save("temp\\vec" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld", "data", θvec_temp[:])

            save("temp\\vec_e" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld", "data", θvec_temp_e[:])
            save("temp\\vec_theta" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld", "data", θvec_temp_θ[:])
            save("temp\\vec_c" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld", "data", θvec_temp_c[:])

        elseif print_ddist > tol * 10.0

            train_gp_result = train_gp(s_inputs_normarlized[:, prediction_sample],
                t_set_normalized[prediction_sample], θvec_guess, 15, 0.0, LL)
            θvec_temp = copy(train_gp_result[2])
            LL = train_gp_result[1]

            train_gp_result_e = train_gp(s_inputs_normarlized[:, prediction_sample],
                e_set_normalized[prediction_sample], θvec_e_history[:, t+1], 15, 0.0, LL_e)
            θvec_temp_e = copy(train_gp_result_e[2])
            LL_e = train_gp_result_e[1]

            train_gp_result_θ = train_gp(s_inputs_normarlized[:, prediction_sample],
                θ_set_normalized[prediction_sample], θvec_θ_history[:, t+1], 15, 0.0, LL_θ)
            θvec_temp_θ = copy(train_gp_result_θ[2])
            LL_θ = train_gp_result_θ[1]

            train_gp_result_c = train_gp(s_inputs_normarlized[:, prediction_sample],
                c_set_normalized[prediction_sample], θvec_c_history[:, t+1], 15, 0.0, LL_c)
            θvec_temp_c = copy(train_gp_result_c[2])
            LL_c = train_gp_result_c[1]

            save("temp\\vec" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld", "data", θvec_temp[:])

            save("temp\\vec_e" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld", "data", θvec_temp_e[:])
            save("temp\\vec_theta" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld", "data", θvec_temp_θ[:])
            save("temp\\vec_c" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld", "data", θvec_temp_c[:])
        end


        @everywhere θvec = load("temp\\vec" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld")["data"]

        @everywhere θvec_e = load("temp\\vec_e" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld")["data"]
        @everywhere θvec_θ = load("temp\\vec_theta" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld")["data"]
        global θvec_history[:, t] = θvec[1:end]
        global θvec_e_history[:, t] = θvec_e[1:end]
        global θvec_θ_history[:, t] = θvec_θ[1:end]

        @everywhere θvec_c = load("temp\\vec_c" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld")["data"]
        global θvec_c_history[:, t] = θvec_c[1:end]

        if t == 1
            train_gp_result_k = train_gp(s_inputs_normarlized[:, prediction_sample],
                k_set_normalized[prediction_sample], θvec, 15, 0.0)
            θvec_temp_k = copy(train_gp_result_k[2])
        end
        if control_LB[4] != 0.0 && t == 1
            train_gp_result_τ_θ = train_gp(s_inputs_normarlized[:, prediction_sample],
                τ_θ_set_normalized[prediction_sample], θvec, 15, 0.0)
            θvec_temp_τ_θ = copy(train_gp_result_τ_θ[2])
        end

        if control_UB[5] != 0.0 && t == 1
            train_gp_result_τ_B = train_gp(s_inputs_normarlized[:, prediction_sample],
                τ_B_set_normalized[prediction_sample], θvec, 15, 0.0)
            θvec_temp_τ_B = copy(train_gp_result_τ_B[2])
        end

        if control_LB[6] != 0.0 && t == 1
            train_gp_result_τ_G = train_gp(s_inputs_normarlized[:, prediction_sample],
                τ_G_set_normalized[prediction_sample], θvec, 15, 0.0)
            θvec_temp_τ_G = copy(train_gp_result_τ_G[2])
        end

        if t == 1
            save("temp\\vec_k" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld", "data", θvec_temp_k[:])
            @everywhere θvec_k = load("temp\\vec_k" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld")["data"]
            # global θvec_k_history[:, t] = θvec_k[1:end]
        end

        if control_LB[4] != 0.0 && t == 1
            save("temp\\vec_τ_theta" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld", "data", θvec_temp_τ_θ[:])
            @everywhere θvec_τ_θ = load("temp\\vec_τ_theta" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld")["data"]
            global θvec_τ_θ_history[:, t] = θvec_τ_θ[1:end]
        end

        if control_UB[5] != 0.0 && t == 1
            save("temp\\vec_τ_B" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld", "data", θvec_temp_τ_B[:])
            @everywhere θvec_τ_B = load("temp\\vec_τ_B" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld")["data"]
            global θvec_τ_B_history[:, t] = θvec_τ_B[1:end]
        end

        if control_LB[6] != 0.0 && t == 1
            save("temp\\vec_τ_G" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld", "data", θvec_temp_τ_G[:])
            @everywhere θvec_τ_G = load("temp\\vec_τ_G" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld")["data"]
            global θvec_τ_G_history[:, t] = θvec_τ_G[1:end]
        end




        @everywhere mK = generate_K(s_inputs_normarlized[:, prediction_sample], θvec)
        @everywhere global va = generate_a(mK, t_set_normalized[prediction_sample])
        @everywhere global gp_para = copy(θvec)

        @everywhere global function fValue_function(X)
            return gp_predict(X, s_inputs_normarlized[:, prediction_sample], va, gp_para, t_mean, t_std, LB, UB)
        end

        @everywhere mK_e = generate_K(s_inputs_normarlized[:, prediction_sample], θvec_e)
        @everywhere global va_e = generate_a(mK_e, e_set_normalized[prediction_sample])
        @everywhere global gp_para_e = copy(θvec_e)

        @everywhere global function e_function(X)
            return gp_predict(X, s_inputs_normarlized[:, prediction_sample], va_e, gp_para_e, e_mean, e_std, LB, UB)
        end

        @everywhere mK_θ = generate_K(s_inputs_normarlized[:, prediction_sample], θvec_θ)
        @everywhere global va_θ = generate_a(mK_θ, θ_set_normalized[prediction_sample])
        @everywhere global gp_para_θ = copy(θvec_θ)

        @everywhere global function θ_function(X)
            return gp_predict(X, s_inputs_normarlized[:, prediction_sample], va_θ, gp_para_θ, θ_mean, θ_std, LB, UB)
        end

        if t == 1
            @everywhere mK_k = generate_K(s_inputs_normarlized[:, prediction_sample], θvec_k)
            @everywhere global va_k = generate_a(mK_k, k_set_normalized[prediction_sample])
            @everywhere global gp_para_k = copy(θvec_k)

            @everywhere global function k_function(X)
                return gp_predict(X, s_inputs_normarlized[:, prediction_sample], va_k, gp_para_k, k_mean, k_std, LB, UB)
            end
        else
            @everywhere global function k_function(X)
                return 0.0
            end
        end


        if control_LB[4] != 0.0 && t == 1
            @everywhere mK_τ_θ = generate_K(s_inputs_normarlized[:, prediction_sample], θvec_τ_θ)
            @everywhere global va_τ_θ = generate_a(mK_τ_θ, τ_θ_set_normalized[prediction_sample])
            @everywhere global gp_para_τ_θ = copy(θvec_τ_θ)

            @everywhere global function τ_θ_function(X)
                return gp_predict(X, s_inputs_normarlized[:, prediction_sample], va_τ_θ, gp_para_τ_θ, τ_θ_mean, τ_θ_std, LB, UB)
            end
        else
            @everywhere global function τ_θ_function(X)
                return 0.0
            end
        end

        if control_UB[5] != 0.0 && t == 1
            @everywhere mK_τ_B = generate_K(s_inputs_normarlized[:, prediction_sample], θvec_τ_B)
            @everywhere global va_τ_B = generate_a(mK_τ_B, τ_B_set_normalized[prediction_sample])
            @everywhere global gp_para_τ_B = copy(θvec_τ_B)

            @everywhere global function τ_B_function(X)
                return gp_predict(X, s_inputs_normarlized[:, prediction_sample], va_τ_B, gp_para_τ_B, τ_B_mean, τ_B_std, LB, UB)
            end
        else
            @everywhere global function τ_B_function(X)
                return 0.0
            end
        end

        if control_LB[6] != 0.0 && t == 1
            @everywhere mK_τ_G = generate_K(s_inputs_normarlized[:, prediction_sample], θvec_τ_G)
            @everywhere global va_τ_G = generate_a(mK_τ_G, τ_G_set_normalized[prediction_sample])
            @everywhere global gp_para_τ_G = copy(θvec_τ_G)

            @everywhere global function τ_G_function(X)
                return gp_predict(X, s_inputs_normarlized[:, prediction_sample], va_τ_G, gp_para_τ_G, τ_G_mean, τ_G_std, LB, UB)
            end
        else
            @everywhere global function τ_G_function(X)
                return 0.0
            end
        end

        @everywhere mK_c = generate_K(s_inputs_normarlized[:, prediction_sample], θvec_c)
        @everywhere global va_c = generate_a(mK_c, c_set_normalized[prediction_sample])
        @everywhere global gp_para_c = copy(θvec_c)
        @everywhere global function c_function(X)
            return gp_predict(X, s_inputs_normarlized[:, prediction_sample], va_c, gp_para_c, c_mean, c_std, LB, UB)
        end


        t_set_history[:, t] = copy(t_set)
        s_tomorrow_history[:, :, t] = copy(s_tomorrow_set)
        control_set_history[:, :, t] = copy(control_set)
        ret_set_history[:, t] = copy(ret_set)

        global predict_values = Array{Float64}(undef, N_inputs)
        global predict_values_e = Array{Float64}(undef, N_inputs)
        global predict_values_θ = Array{Float64}(undef, N_inputs)
        global predict_values_k = Array{Float64}(undef, N_inputs)
        global predict_values_τ_θ = Array{Float64}(undef, N_inputs)
        global predict_values_τ_B = Array{Float64}(undef, N_inputs)
        global predict_values_τ_G = Array{Float64}(undef, N_inputs)
        global predict_values_c = Array{Float64}(undef, N_inputs)

        for i = 1:(N_inputs)
            predict_value = fValue_function(s_inputs[:, i])
            predict_values[i] = predict_value

            predict_values_e[i] = e_function(s_inputs[:, i])
            predict_values_θ[i] = θ_function(s_inputs[:, i])
            predict_values_k[i] = k_function(s_inputs[:, i])
            predict_values_τ_θ[i] = τ_θ_function(s_inputs[:, i])
            predict_values_τ_B[i] = τ_B_function(s_inputs[:, i])
            predict_values_τ_G[i] = τ_G_function(s_inputs[:, i])
            predict_values_c[i] = c_function(s_inputs[:, i])
            var_set[i] = fPosterior_variance(bound_normalize(s_tomorrow_set[:, i], LB, UB), s_inputs_normarlized[:, prediction_sample], mK, θvec)
        end
        global var_set_history[:, t] = copy(var_set)

        global prediction_loss = abs.(predict_values .- t_set)
        global prediction_loss_e = abs.(predict_values_e .- control_set[1, :])
        global prediction_loss_θ = abs.(predict_values_θ .- control_set[2, :])
        global prediction_loss_k = abs.(predict_values_k .- control_set[3, :])
        global prediction_loss_τ_θ = abs.(predict_values_τ_θ .- control_set[3, :])
        global prediction_loss_τ_B = abs.(predict_values_τ_B .- control_set[4, :])
        global prediction_loss_τ_G = abs.(predict_values_τ_G .- control_set[5, :])
        global prediction_loss_c = abs.(predict_values_c .- c_set[:])




        println([t, mean(prediction_loss[findall(ret_set .== 1.0)]), maximum(prediction_loss[findall(ret_set .== 1.0)]),
            mean(prediction_loss_e[findall(ret_set .== 1.0)]), maximum(prediction_loss_e[findall(ret_set .== 1.0)]),
            mean(prediction_loss_θ[findall(ret_set .== 1.0)]), maximum(prediction_loss_θ[findall(ret_set .== 1.0)]),
            mean(prediction_loss_k[findall(ret_set .== 1.0)]), maximum(prediction_loss_k[findall(ret_set .== 1.0)]),
            mean(prediction_loss_c[findall(ret_set .== 1.0)]), maximum(prediction_loss_c[findall(ret_set .== 1.0)]),
            mean(prediction_loss_τ_θ[findall(ret_set .== 1.0)]), maximum(prediction_loss_τ_θ[findall(ret_set .== 1.0)]),
            mean(prediction_loss_τ_B[findall(ret_set .== 1.0)]), maximum(prediction_loss_τ_B[findall(ret_set .== 1.0)]),
            mean(prediction_loss_τ_G[findall(ret_set .== 1.0)]), maximum(prediction_loss_τ_G[findall(ret_set .== 1.0)]),
            mean(abs.(loss_1_1_set[findall(ret_set .== 1.0)])), mean(abs.(loss_1_2_set[findall(ret_set .== 1.0)])),
            mean(abs.(loss_2_1_set[findall(ret_set .== 1.0)])), mean(abs.(loss_2_2_set[findall(ret_set .== 1.0)])),
            maximum(abs.(loss_1_1_set[findall(ret_set .== 1.0)])), maximum(abs.(loss_1_2_set[findall(ret_set .== 1.0)])),
            maximum(abs.(loss_2_1_set[findall(ret_set .== 1.0)])), maximum(abs.(loss_2_2_set[findall(ret_set .== 1.0)])),
            print_ddist, fValue_function(s_initial)])
        println(control_set[:, dimension_input])
        println(control_set[:, 1])
        if (θvec[end] > -6)
            t = 0
            policy_iter = 100
        end

        if print_ddist < tol
            t = 1
        else
            t = t - 1
        end

        # break
    end


    @everywhere result = simulation()

    if policy_iter == 1
        Results = result
    else
        Results = [Results; result]
    end
    save("temp\\Results_" * regimes[policy_iter] * "Medium_SR.jld", "data", Results)
end


M_table = Array{Float64}(undef, 4)
CEV_table = Array{Float64}(undef, 4)

table_front = 1
result_length = size(result)[1]
table_end = result_length
V_t_BAU = Results[14, 1]
for i = 1:4
    table = Results[table_front:table_end, :]

    M_table[i] = table[8, 85]
    CEV_table[i] = (table[14, 1] / V_t_BAU)^(1.0 / (1.0 - α)) - 1.0

    table_front = table_front + result_length
    table_end = table_end + result_length
end

s_t = s_inputs[:, dimension_input]
x_t_B = control_set[:, dimension_input]
s_t_history = Array{Float64}(undef, D, T)
x_t_history = Array{Float64}(undef, length(control_LB), T)
c_t_history = Array{Float64}(undef, 1, T)
i_t_history = Array{Float64}(undef, 1, T)
SCC_t_history = Array{Float64}(undef, 1, T)
eB_t_history = Array{Float64}(undef, 1, T)
V_t_history = Array{Float64}(undef, 1, T)
y_t_history = Array{Float64}(undef, 1, T)

if true
    
    x_t_history = Array{Float64}(undef, length(control_LB), T)

    c_t_history = Array{Float64}(undef, 1, T)
    # SCC_t_history = Array{Float64}(undef, 1, T)
    eB_t_history = Array{Float64}(undef, 1, T)
    # println()

    for t = 1:T
        s_t_history[:, t] = s_t
        # V_t_history[t] = fValue_function(s_t)
        vfi_result = value_function_iteration(x_t_B, s_t)
        x_t_B = copy(vfi_result[2])
        println(vfi_result)
        println(x_t_B)

        μ_t = energy_price_index(x_t_B, s_t)
        SCC_t = x_t_B[5]
        tau_G = pG_0 * (pB / (pB + SCC_t) - 1.0)
        tau_G = (pG_0 + tau_G) / pG_0 - 1.0


        x_t_G = [control_UB[1], control_UB[2], control_UB[3], 0.0, 0.0, tau_G]
        
        p_t = energy_price_index(x_t_G, s_t)

        τ_θ = -exp(s_t[3])^σ * (μ_t - p_t) / (total_consumption(x_t_B, s_t))^(-α) / exp(θ_function(s_t))^2.0


        x_t_G = [control_UB[1], control_UB[2], control_UB[3], τ_θ, 0.0, tau_G]

        c_t_history[t] = total_consumption(x_t_B, s_t)
        eB_t_history[t] = eB_constraint(x_t_B, s_t) * 7.5 * 3.667

        s_t = law_of_motion(x_t_B, s_t)
        SCC_t_history[t] = -β *
                           ForwardDiff.gradient(fValue_function, s_t)[2] / (c_t_history[t]^(-α)) .* ζ_0

        x_t_history[:, t] = x_t_G
        # println(s_t)
        # println(x_t)
        # println(t)
        # println()

    end

end