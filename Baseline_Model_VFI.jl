function vfi(control_LB, control_UB, fValue_function, e_function, θ_function, c_function)

    @eval fV_prior(X) = -0.0 # Our trival prior guess; notice X is a vector here
    va = 0.0
    gp_para = 0.0
    @eval fValue_function(s) = fV_prior(s)
    @eval e_function(s) = 1.0
    @eval θ_function(s) = s[1]
    @eval k_function(s) = s[3]
    @eval τ_θ_function(s) = 0.0
    @eval τ_B_function(s) = 0.0
    @eval τ_G_function(s) = 0.0
    @eval c_function(s) = 1.0

    global t_set = SharedArray{Float64}(N_inputs)
    global control_set = SharedArray{Float64}(length(control_LB), N_inputs)
    global var_set = SharedArray{Float64}(N_inputs)
    global c_set = SharedArray{Float64}(N_inputs)
    global s_tomorrow_set = SharedArray{Float64}(D, N_inputs)
    global ret_set = SharedArray{Float64}(N_inputs)
    global loss_1_1_set = SharedArray{Float64}(N_inputs)
    global loss_1_2_set = SharedArray{Float64}(N_inputs)
    global loss_2_1_set = SharedArray{Float64}(N_inputs)
    global loss_2_2_set = SharedArray{Float64}(N_inputs)
    global LOO = Array{Float64}(undef, N_training_inputs)

    # global s_inputs_history = Array{Float64}(undef, D, N_inputs, max_iter)
    global t_set_history = Array{Float64}(undef, N_inputs, max_iter)
    global t_set_normalized_history = Array{Float64}(undef, N_inputs, max_iter)
    global s_tomorrow_history = Array{Float64}(undef, D, N_inputs, max_iter)
    global control_set_history = Array{Float64}(undef, length(control_LB), N_inputs, max_iter)
    global loss_1_1_set_history = Array{Float64}(undef, N_inputs, max_iter)
    global loss_1_2_set_history = Array{Float64}(undef, N_inputs, max_iter)
    global loss_2_1_set_history = Array{Float64}(undef, N_inputs, max_iter)
    global loss_2_2_set_history = Array{Float64}(undef, N_inputs, max_iter)
    global ret_set_history = Array{Float64}(undef, N_inputs, max_iter)
    global var_set_history = Array{Float64}(undef, N_inputs, max_iter)

    global θvec_history = Array{Float64}(undef, kernel_length + 1, max_iter)
    global θvec_e_history = Array{Float64}(undef, kernel_length + 1, max_iter)
    global θvec_θ_history = Array{Float64}(undef, kernel_length + 1, max_iter)
    global θvec_τ_θ_history = Array{Float64}(undef, kernel_length + 1, max_iter)
    global θvec_τ_B_history = Array{Float64}(undef, kernel_length + 1, max_iter)
    global θvec_τ_G_history = Array{Float64}(undef, kernel_length + 1, max_iter)
    global θvec_c_history = Array{Float64}(undef, kernel_length + 1, max_iter)

    global LOO_history = Array{Float64}(undef, N_inputs, max_iter)

    t = max_iter
    print_ddist = 10.0
    LL = 0.0
    LL_e = 0.0
    LL_θ = 0.0
    LL_c = 0.0

    while t >= 1
        println(t)
        time_start = time()
        # s_inputs_history[:, :, t] = s_inputs
        # t_set_hybrid_raw = Array{Float64}(undef, optim_size)

        @sync @distributed for i = 1:N_inputs
            s_today = s_inputs[:, i]
            if t > max_iter - 1
                vfi_result = value_function_iteration_no_constraint(
                    [log(2.3), s_today[1], k_LB, 0.0, 0.0, 0.0], s_today)
                println(vfi_result)
                convergence = 1.0
                # elseif t == max_iter - 1
                #     control_init = value_function_iteration_no_constraint(
                #         [log(2.3), s_today[1], k_LB, 0.0, 0.0, 0.0], s_today)[2]
                #     convergence = 0.0
                #     vfi_result = value_function_iteration(control_init, s_today)
                #     convergence = (vfi_result[3] == "XTOL_REACHED")

            else
                control_init = control_set[:, i]
                convergence = 0.0
                vfi_result = value_function_iteration(control_init, s_today)
                convergence = (vfi_result[3] == "XTOL_REACHED") &&
                              (abs(y_tech_forward_constraint(vfi_result[2], s_today)) < 1e-4) &&
                              (abs(capital_euler_constraint(vfi_result[2], s_today)) < 1e-4)
                # println([i, control_init, vfi_result[3], vfi_result[2]])
            end

            # if convergence != 1
            #     control_init = control_set_history[:, i, t+1]
            #     control_init[4:6] .= 0.0
            #     vfi_result = value_function_iteration(control_init, s_today)
            #     convergence = (vfi_result[3] == "XTOL_REACHED")

            # end

            if convergence != 1
                # println(i)
                control_init = value_function_iteration_no_constraint(
                    [log(2.3), s_today[1], k_LB, 0.0, 0.0, 0.0], s_today)[2]
                vfi_result = value_function_iteration(control_init, s_today)
                convergence = (vfi_result[3] == "XTOL_REACHED")

            end
            t_set[i] = -1.0 * copy(vfi_result[1])
            c_set[i] = total_consumption(vfi_result[2], s_today)

            control_set[:, i] = copy(vfi_result[2])
            s_tomorrow = law_of_motion(control_set[:, i], s_today)
            s_tomorrow_set[:, i] = s_tomorrow
            ret_set[i] = convergence

            if convergence != 1
                println(i)
                control_set[:, i] = copy(control_set_history[:, i, t+1])
                t_set[i] = copy(fValue_today(control_set_history[:, i, t+1], s_today))
            end


        end

        if (minimum(control_set[2, 1:optim_size]) <= θ_y_LB) ||
           (maximum(control_set[2, 1:optim_size]) >= θ_y_UB) ||
           (maximum(s_tomorrow_set[2, 1:optim_size]) >= M_UB) ||
           (maximum(control_set[1, 1:optim_size]) >= e_UB)
            # println("break")
            break
        end

        # if minimum(ret_set) == 0
        #     break
        # end

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

                # println(i)
                # println(y_tech_forward_constraint(x_t, s_t))
                # println(y_tech_forward_constraint_test(x_t, s_t, x_t_2))
                try
                    loss_1_1_set[i] = y_tech_forward_constraint(x_t, s_t) * convergence_1 * convergence_2
                    loss_1_2_set[i] = capital_euler_constraint(x_t, s_t) * convergence_1 * convergence_2
                    loss_2_1_set[i] = y_tech_forward_constraint_test(x_t, s_t, x_t_2) * convergence_1 * convergence_2
                    loss_2_2_set[i] = capital_euler_constraint_test(x_t, s_t, x_t_2) * convergence_1 * convergence_2
                catch
                    println("constraint error")
                    println(i)
                end
                # println()
                # println(x_t_2[1:2])
                # println([e_function(s_t_2), θ_function(s_t_2)])
                # println(total_consumption(x_t_2, s_t_2))
                # println(c_function(s_t_2))
                # if y_tech_forward_constraint(x_t, s_t) > max_tol
                #     max_tol = y_tech_forward_constraint(x_t, s_t)
                #     println(i)
                # end
                # if abs(y_tech_forward_constraint(x_t, s_t)) > 1e-4
                #     break
                # end
            end
            # if max_tol > 1e-6
            #     break
            # end
            # println(maximum(abs.(loss_1_set)))
            loss_1_1_set_history[:, t] = copy(loss_1_1_set)
            loss_1_2_set_history[:, t] = copy(loss_1_2_set)
            loss_2_1_set_history[:, t] = copy(loss_2_1_set)
            loss_2_2_set_history[:, t] = copy(loss_2_2_set)
        end


        global t_set_normalized = copy(t_set)
        global t_max = copy(maximum(t_set))
        global t_min = copy(minimum(t_set))
        global t_mean = mean(t_set_normalized)
        global t_std = std(t_set_normalized)
        global t_set_normalized = std_normalize.(t_set_normalized, t_mean, t_std)

        global e_set_normalized = copy(control_set[1, :])
        global e_mean = mean(e_set_normalized)
        global e_std = std(e_set_normalized)
        global e_set_normalized = std_normalize.(e_set_normalized, e_mean, e_std)


        global c_set_normalized = copy(c_set[:])
        global c_mean = mean(c_set_normalized)
        global c_std = std(c_set_normalized)
        global c_set_normalized = std_normalize.(c_set_normalized, c_mean, c_std)


        global θ_set_normalized = copy(control_set[2, :])
        global θ_mean = mean(θ_set_normalized)
        global θ_std = std(θ_set_normalized)
        global θ_set_normalized = std_normalize.(θ_set_normalized, θ_mean, θ_std)

        global k_set_normalized = copy(control_set[3, :])
        global k_mean = mean(k_set_normalized)
        global k_std = std(k_set_normalized)
        global k_set_normalized = std_normalize.(k_set_normalized, k_mean, k_std)


        global τ_θ_set_normalized = copy(control_set[4, :])
        global τ_θ_mean = mean(τ_θ_set_normalized)
        global τ_θ_std = std(τ_θ_set_normalized)
        global τ_θ_set_normalized = std_normalize.(τ_θ_set_normalized, τ_θ_mean, τ_θ_std)

        global τ_B_set_normalized = copy(control_set[5, :])
        global τ_B_mean = mean(τ_B_set_normalized)
        global τ_B_std = std(τ_B_set_normalized)
        global τ_B_set_normalized = std_normalize.(τ_B_set_normalized, τ_B_mean, τ_B_std)

        global τ_G_set_normalized = copy(control_set[6, :])
        global τ_G_mean = mean(τ_G_set_normalized)
        global τ_G_std = std(τ_G_set_normalized)
        global τ_G_set_normalized = std_normalize.(τ_G_set_normalized, τ_G_mean, τ_G_std)


        if t < max_iter - 1

            print_ddist = maximum(abs.(t_set .- t_set_history[:, t+1]))
            println(print_ddist)
        end


        if t == max_iter
            # global θvec_guess = [25.538198478076357, 28.899412431190584, 21.780221777517266, -9.011236053406305, -22.938207767834484]
            global θvec_guess = vcat(0.1 * ones(kernel_length), -8.0)
        else
            global θvec_guess = copy(θvec_history[:, t+1])
        end

        if t == max_iter
            println(s_inputs_normarlized[:, 1:optim_size])
            println(t_set[1:optim_size])
            println(t_set_normalized[1:optim_size])
            global train_gp_result = train_gp(s_inputs_normarlized[:, 1:optim_size],
                t_set_normalized[1:optim_size], θvec_guess, 30, 0.0, LL)
            θvec_temp = copy(train_gp_result[2])
            LL = train_gp_result[1]

            global train_gp_result_e = train_gp(s_inputs_normarlized[:, 1:optim_size],
                e_set_normalized[1:optim_size], θvec_temp, 30, 0.0, LL_e)
            θvec_temp_e = copy(train_gp_result_e[2])
            LL_e = train_gp_result_e[1]

            global train_gp_result_θ = train_gp(s_inputs_normarlized[:, 1:optim_size],
                θ_set_normalized[1:optim_size], θvec_temp, 30, 0.0, LL_θ)
            θvec_temp_θ = copy(train_gp_result_θ[2])
            LL_θ = train_gp_result_θ[1]

            global train_gp_result_c = train_gp(s_inputs_normarlized[:, 1:optim_size],
                c_set_normalized[1:optim_size], θvec_temp, 30, 0.0, LL_c)
            θvec_temp_c = copy(train_gp_result_c[2])
            LL_c = train_gp_result_c[1]

            save("temp\\vec" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld", "data", θvec_temp[:])

            save("temp\\vec_e" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld", "data", θvec_temp_e[:])
            save("temp\\vec_theta" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld", "data", θvec_temp_θ[:])
            save("temp\\vec_c" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld", "data", θvec_temp_c[:])

        elseif print_ddist > tol * 10.0

            global train_gp_result = train_gp(s_inputs_normarlized[:, 1:optim_size],
                t_set_normalized[1:optim_size], θvec_guess, 15, 0.0, LL)
            θvec_temp = copy(train_gp_result[2])
            LL = train_gp_result[1]

            global train_gp_result_e = train_gp(s_inputs_normarlized[:, 1:optim_size],
                e_set_normalized[1:optim_size], θvec_e_history[:, t+1], 15, 0.0, LL_e)
            θvec_temp_e = copy(train_gp_result_e[2])
            LL_e = train_gp_result_e[1]

            global train_gp_result_θ = train_gp(s_inputs_normarlized[:, 1:optim_size],
                θ_set_normalized[1:optim_size], θvec_θ_history[:, t+1], 15, 0.0, LL_θ)
            θvec_temp_θ = copy(train_gp_result_θ[2])
            LL_θ = train_gp_result_θ[1]

            global train_gp_result_c = train_gp(s_inputs_normarlized[:, 1:optim_size],
                c_set_normalized[1:optim_size], θvec_c_history[:, t+1], 15, 0.0, LL_c)
            θvec_temp_c = copy(train_gp_result_c[2])
            LL_c = train_gp_result_c[1]

            save("temp\\vec" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld", "data", θvec_temp[:])

            save("temp\\vec_e" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld", "data", θvec_temp_e[:])
            save("temp\\vec_theta" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld", "data", θvec_temp_θ[:])
            save("temp\\vec_c" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld", "data", θvec_temp_c[:])
        end





        # try

        #     println([compute_marginal_likelihood(s_inputs_normarlized, t_set_normalized, θvec_guess),
        #         compute_marginal_likelihood(s_inputs_normarlized, t_set_normalized, θvec_temp[:])])

        #     if (compute_marginal_likelihood(s_inputs_normarlized, t_set_normalized, θvec_guess) <
        #         compute_marginal_likelihood(s_inputs_normarlized, t_set_normalized, θvec_temp[:]))
        #         save("temp\\vec"*Dates.format(dt, "yyyymmddHHMMSS")*".jld", "data", θvec_temp[:])

        #         θvec = load("temp\\vec"*Dates.format(dt, "yyyymmddHHMMSS")*".jld")["data"]

        #     else
        #         println("keep the θvec")
        #     end
        # catch
        #     println("test failed")
        #     save("temp\\vec"*Dates.format(dt, "yyyymmddHHMMSS")*".jld", "data", θvec_temp[:])
        #     θvec = load("temp\\vec.jld")["data"]
        # end

        θvec = load("temp\\vec" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld")["data"]

        θvec_e = load("temp\\vec_e" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld")["data"]
        θvec_θ = load("temp\\vec_theta" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld")["data"]
        global θvec_history[:, t] = θvec[1:end]
        global θvec_e_history[:, t] = θvec_e[1:end]
        global θvec_θ_history[:, t] = θvec_θ[1:end]

        θvec_c = load("temp\\vec_c" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld")["data"]
        global θvec_c_history[:, t] = θvec_c[1:end]

        if t == 1
            global train_gp_result_k = train_gp(s_inputs_normarlized[:, 1:optim_size],
                k_set_normalized[1:optim_size], θvec, 15, 0.0)
            θvec_temp_k = copy(train_gp_result_k[2])
        end
        if control_LB[4] != 0.0 && t == 1
            global train_gp_result_τ_θ = train_gp(s_inputs_normarlized[:, 1:optim_size],
                τ_θ_set_normalized[1:optim_size], θvec, 15, 0.0)
            θvec_temp_τ_θ = copy(train_gp_result_τ_θ[2])
        end

        if control_UB[5] != 0.0 && t == 1
            global train_gp_result_τ_B = train_gp(s_inputs_normarlized[:, 1:optim_size],
                τ_B_set_normalized[1:optim_size], θvec, 15, 0.0)
            θvec_temp_τ_B = copy(train_gp_result_τ_B[2])
        end

        if control_LB[6] != 0.0 && t == 1
            global train_gp_result_τ_G = train_gp(s_inputs_normarlized[:, 1:optim_size],
                τ_G_set_normalized[1:optim_size], θvec, 15, 0.0)
            θvec_temp_τ_G = copy(train_gp_result_τ_G[2])
        end

        if t == 1
            save("temp\\vec_k" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld", "data", θvec_temp_k[:])
            θvec_k = load("temp\\vec_k" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld")["data"]
            # global θvec_k_history[:, t] = θvec_k[1:end]
        end

        if control_LB[4] != 0.0 && t == 1
            save("temp\\vec_τ_theta" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld", "data", θvec_temp_τ_θ[:])
            θvec_τ_θ = load("temp\\vec_τ_theta" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld")["data"]
            global θvec_τ_θ_history[:, t] = θvec_τ_θ[1:end]
        end

        if control_UB[5] != 0.0 && t == 1
            save("temp\\vec_τ_B" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld", "data", θvec_temp_τ_B[:])
            θvec_τ_B = load("temp\\vec_τ_B" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld")["data"]
            global θvec_τ_B_history[:, t] = θvec_τ_B[1:end]
        end

        if control_LB[6] != 0.0 && t == 1
            save("temp\\vec_τ_G" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld", "data", θvec_temp_τ_G[:])
            θvec_τ_G = load("temp\\vec_τ_G" * Dates.format(dt, "yyyymmddHHMMSS") * ".jld")["data"]
            global θvec_τ_G_history[:, t] = θvec_τ_G[1:end]
        end




        mK = generate_K(s_inputs_normarlized[:, 1:optim_size], θvec)
        global va = generate_a(mK, t_set_normalized[1:optim_size])
        global gp_para = copy(θvec)

        @eval function fValue_function(X)
            return gp_predict(X, s_inputs_normarlized[:, 1:optim_size], va, gp_para, t_mean, t_std, LB, UB)
        end

        mK_e = generate_K(s_inputs_normarlized[:, 1:optim_size], θvec_e)
        global va_e = generate_a(mK_e, e_set_normalized[1:optim_size])
        global gp_para_e = copy(θvec_e)

        @eval function e_function(X)
            return gp_predict(X, s_inputs_normarlized[:, 1:optim_size], va_e, gp_para_e, e_mean, e_std, LB, UB)
        end

        mK_θ = generate_K(s_inputs_normarlized[:, 1:optim_size], θvec_θ)
        global va_θ = generate_a(mK_θ, θ_set_normalized[1:optim_size])
        global gp_para_θ = copy(θvec_θ)

        @eval function θ_function(X)
            return gp_predict(X, s_inputs_normarlized[:, 1:optim_size], va_θ, gp_para_θ, θ_mean, θ_std, LB, UB)
        end

        if t == 1
            mK_k = generate_K(s_inputs_normarlized[:, 1:optim_size], θvec_k)
            global va_k = generate_a(mK_k, k_set_normalized[1:optim_size])
            global gp_para_k = copy(θvec_k)

            @eval function k_function(X)
                return gp_predict(X, s_inputs_normarlized[:, 1:optim_size], va_k, gp_para_k, k_mean, k_std, LB, UB)
            end
        else
            @eval function k_function(X)
                return 0.0
            end
        end


        if control_LB[4] != 0.0 && t == 1
            mK_τ_θ = generate_K(s_inputs_normarlized[:, 1:optim_size], θvec_τ_θ)
            global va_τ_θ = generate_a(mK_τ_θ, τ_θ_set_normalized[1:optim_size])
            global gp_para_τ_θ = copy(θvec_τ_θ)

            @eval function τ_θ_function(X)
                return gp_predict(X, s_inputs_normarlized[:, 1:optim_size], va_τ_θ, gp_para_τ_θ, τ_θ_mean, τ_θ_std, LB, UB)
            end
        else
            @eval function τ_θ_function(X)
                return 0.0
            end
        end

        if control_UB[5] != 0.0 && t == 1
            mK_τ_B = generate_K(s_inputs_normarlized[:, 1:optim_size], θvec_τ_B)
            global va_τ_B = generate_a(mK_τ_B, τ_B_set_normalized[1:optim_size])
            global gp_para_τ_B = copy(θvec_τ_B)

            @eval function τ_B_function(X)
                return gp_predict(X, s_inputs_normarlized[:, 1:optim_size], va_τ_B, gp_para_τ_B, τ_B_mean, τ_B_std, LB, UB)
            end
        else
            @eval function τ_B_function(X)
                return 0.0
            end
        end

        if control_LB[6] != 0.0 && t == 1
            mK_τ_G = generate_K(s_inputs_normarlized[:, 1:optim_size], θvec_τ_G)
            global va_τ_G = generate_a(mK_τ_G, τ_G_set_normalized[1:optim_size])
            global gp_para_τ_G = copy(θvec_τ_G)

            @eval function τ_G_function(X)
                return gp_predict(X, s_inputs_normarlized[:, 1:optim_size], va_τ_G, gp_para_τ_G, τ_G_mean, τ_G_std, LB, UB)
            end
        else
            @eval function τ_G_function(X)
                return 0.0
            end
        end

        mK_c = generate_K(s_inputs_normarlized[:, 1:optim_size], θvec_c)
        global va_c = generate_a(mK_c, c_set_normalized[1:optim_size])
        global gp_para_c = copy(θvec_c)
        @eval function c_function(X)
            return gp_predict(X, s_inputs_normarlized[:, 1:optim_size], va_c, gp_para_c, c_mean, c_std, LB, UB)
        end


        # s_inputs_history[:, :, t] = copy(s_inputs)
        t_set_history[:, t] = copy(t_set)
        t_set_normalized_history[:, t] = copy(t_set_normalized)
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
            var_set[i] = fPosterior_variance(bound_normalize(s_tomorrow_set[:, i], LB, UB), s_inputs_normarlized[:, 1:optim_size], mK, θvec)
        end
        global var_set_history = Array{Float64}(undef, N_inputs, max_iter)

        global prediction_loss = abs.(predict_values .- t_set) ./
                                 (maximum(t_set) - minimum(t_set))
        global prediction_loss_e = abs.(predict_values_e .- control_set[1, :])
        global prediction_loss_θ = abs.(predict_values_θ .- control_set[2, :])
        global prediction_loss_k = abs.(predict_values_k .- control_set[3, :])
        global prediction_loss_τ_θ = abs.(predict_values_τ_θ .- control_set[3, :])
        global prediction_loss_τ_B = abs.(predict_values_τ_B .- control_set[4, :])
        global prediction_loss_τ_G = abs.(predict_values_τ_G .- control_set[5, :])
        global prediction_loss_c = abs.(predict_values_c .- c_set[:])




        println([t, mean(prediction_loss), maximum(prediction_loss),
            mean(prediction_loss_e), maximum(prediction_loss_e),
            mean(prediction_loss_θ), maximum(prediction_loss_θ),
            mean(prediction_loss_k), maximum(prediction_loss_k),
            mean(prediction_loss_c), maximum(prediction_loss_c),
            mean(prediction_loss_τ_θ), maximum(prediction_loss_τ_θ),
            mean(prediction_loss_τ_B), maximum(prediction_loss_τ_B),
            mean(prediction_loss_τ_G), maximum(prediction_loss_τ_G),
            mean(abs.(loss_1_1_set)), mean(abs.(loss_1_2_set)),
            mean(abs.(loss_2_1_set)), mean(abs.(loss_2_2_set)),
            maximum(abs.(loss_1_1_set)), maximum(abs.(loss_1_2_set)),
            maximum(abs.(loss_2_1_set)), maximum(abs.(loss_2_2_set)),
            print_ddist, fValue_function(s_initial)])
        println(control_set[:, dimension_input])
        println(control_set[:, 1])
        # if (maximum(abs.(loss_2_1_set[1:optim_size])) > 1e-1) || (maximum(abs.(loss_2_2_set[1:optim_size])) > 1e-1)
        #     break
        # end

        if print_ddist < tol
            t = 1
        else
            t = t - 1
        end
        # break
    end


    s_t = s_inputs[:, dimension_input]
    x_t = control_set[:, dimension_input]
    s_t_history = Array{Float64}(undef, D, T)
    x_t_history = Array{Float64}(undef, length(control_LB), T)
    c_t_history = Array{Float64}(undef, 1, T)
    i_t_history = Array{Float64}(undef, 1, T)
    SCC_t_history = Array{Float64}(undef, 1, T)
    eB_t_history = Array{Float64}(undef, 1, T)
    println()

    for t = 1:T
        s_t_history[:, t] = s_t
        x_t = [e_function(s_t), θ_function(s_t), k_function(s_t),
            τ_θ_function(s_t), τ_B_function(s_t), τ_G_function(s_t)]
        # println(goods_transition_cost(x_t, s_t))
        # if t > 1
        #     println(y_tech_forward_constraint_test(x_t_history[:, t-1], s_t_history[:, t-1], x_t))
        #     println(capital_euler_constraint_test(x_t_history[:, t-1], s_t_history[:, t-1], x_t))
        #     println()
        # end
        c_t_history[t] = total_consumption(x_t, s_t)
        i_t_history[t] = (exp(x_t[3]) - (1.0 - δ) * exp(s_t[3])) / (total_consumption(x_t, s_t) + exp(x_t[3]) - (1.0 - δ) * exp(s_t[3]))
        eB_t_history[t] = eB_constraint(x_t, s_t) * 7.5 * 3.667

        s_t = law_of_motion(x_t, s_t)
        SCC_t_history[t] = -β * ForwardDiff.gradient(fValue_function, s_t)[2] / (c_t_history[t]^(-α)) .* ζ_0

        x_t_history[:, t] = x_t
        # println(s_t)
        # println(x_t)
        # println(t)
        # println()

    end
    V_t = 0.0

    for t = 1:T
        V_t = V_t + β^(t-1) * c_t_history[t]^(1.0 - α) / (1.0 - α)
    end

    results = vcat(x_t_history, s_t_history, c_t_history, y_t_history, SCC_t_history, eB_t_history, V_t_history)


    return results
end