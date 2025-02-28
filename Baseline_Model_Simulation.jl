function simulation()

    s_t = s_inputs[:, dimension_input]
    x_t = control_set[:, dimension_input]
    s_t_history = Array{Float64}(undef, D, T)
    x_t_history = Array{Float64}(undef, length(control_LB), T)
    c_t_history = Array{Float64}(undef, 1, T)
    i_t_history = Array{Float64}(undef, 1, T)
    SCC_t_history = Array{Float64}(undef, 1, T)
    eB_t_history = Array{Float64}(undef, 1, T)
    V_t_history = Array{Float64}(undef, 1, T)
    y_t_history = Array{Float64}(undef, 1, T)
    println()

    for t = 1:T
        s_t_history[:, t] = s_t
        x_t = [e_function(s_t), θ_function(s_t), k_function(s_t),
            τ_θ_function(s_t), τ_B_function(s_t), τ_G_function(s_t)]

        c_t_history[t] = total_consumption(x_t, s_t)
        i_t_history[t] = (exp(x_t[3]) - (1.0 - δ) * exp(s_t[3])) / (total_consumption(x_t, s_t) + exp(x_t[3]) - (1.0 - δ) * exp(s_t[3]))
        eB_t_history[t] = eB_constraint(x_t, s_t)
        y_t_history[t] = raw_goods_production(x_t, s_t)
        V_t_history[t] = fValue_function(s_t)
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
        V_t = V_t + β^(t - 1) * c_t_history[t]^(1.0 - α) / (1.0 - α)
    end

    return results = vcat(x_t_history, s_t_history, c_t_history, y_t_history, SCC_t_history, eB_t_history, V_t_history)

end