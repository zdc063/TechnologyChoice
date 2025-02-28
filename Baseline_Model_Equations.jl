function final_energy_production(x, s)
    e = exp(x[1])

    θ_y = exp(x[2])
    k_prime = exp(x[3])
    τ_y = x[4] / 1.0
    τ_B = x[5] / 1.0
    τ_G = x[6] / 1.0
    # τ_G = (pG_0 + x[6]) / pG_0 - 1.0

    θ_y_bar = exp(s[1])
    M = s[2]
    k = exp(s[3])
    # pG = exp(s[4])
    # ζ = exp(s[5])
    pG = pG_0
    ζ = ζ_0

    eB = eB_constraint(x, s)
    eG = eG_constraint(x, s)
    e = (ω_B * (eB)^ρ_e + ω_G * eG^ρ_e)^(1.0 / ρ_e)
    return e
end


function raw_goods_production(x, s)
    e = exp(x[1])
    θ_y = exp(x[2])
    k_prime = exp(x[3])
    τ_y = x[4] / 1.0
    τ_B = x[5] / 1.0
    τ_G = x[6] / 1.0
    # τ_G = (pG_0 + x[6]) / pG_0 - 1.0

    θ_y_bar = exp(s[1])
    M = s[2]
    k = exp(s[3])
    # pG = exp(s[4])
    # ζ = exp(s[5])
    pG = pG_0
    ζ = ζ_0

    y_tilde = ((θ_y^(α_y) * e)^ρ_y + (θ_y^(α_y - 1.0) * k^σ)^ρ_y)^(1.0 / ρ_y)

    return y_tilde
end

function goods_transition_cost(x, s)
    e = exp(x[1])
    θ_y = exp(x[2])
    k_prime = exp(x[3])
    τ_y = x[4] / 1.0
    τ_B = x[5] / 1.0
    τ_G = x[6] / 1.0
    # τ_G = (pG_0 + x[6]) / pG_0 - 1.0

    θ_y_bar = exp(s[1])
    M = s[2]
    k = exp(s[3])
    # pG = exp(s[4])
    # ζ = exp(s[5])
    pG = pG_0
    ζ = ζ_0

    Ψ_y = exp(-0.5 * γ_y * (θ_y / θ_y_bar - 1.0)^2.0)
    return Ψ_y
end

function derivative_goods_transition_cost(x, s)
    e = exp(x[1])
    θ_y = exp(x[2])
    k_prime = exp(x[3])
    τ_y = x[4] / 1.0
    τ_B = x[5] / 1.0
    τ_G = x[6] / 1.0
    # τ_G = (pG_0 + x[6]) / pG_0 - 1.0

    θ_y_bar = exp(s[1])
    M = s[2]
    k = exp(s[3])
    # pG = exp(s[4])
    # ζ = exp(s[5])
    pG = pG_0
    ζ = ζ_0

    dΨ_y_over_ratio = -γ_y * (θ_y / θ_y_bar - 1.0) *
                      exp(-0.5 * γ_y * (θ_y / θ_y_bar - 1.0)^2.0)
    return dΨ_y_over_ratio
end


function damage_function(x, s)
    e = exp(x[1])
    θ_y = exp(x[2])
    k_prime = exp(x[3])
    τ_y = x[4] / 1.0
    τ_B = x[5] / 1.0
    τ_G = x[6] / 1.0
    # τ_G = (pG_0 + x[6]) / pG_0 - 1.0

    θ_y_bar = exp(s[1])
    M = s[2]
    k = exp(s[3])
    # pG = exp(s[4])
    # ζ = exp(s[5])
    pG = pG_0
    ζ = ζ_0

    damage = 1.0 - d * (1.6 * M)^2.0
    return damage
end

function damage_function_derivative(x, s)
    e = exp(x[1])
    θ_y = exp(x[2])
    k_prime = exp(x[3])
    τ_y = x[4] / 1.0
    τ_B = x[5] / 1.0
    τ_G = x[6] / 1.0
    # τ_G = (pG_0 + x[6]) / pG_0 - 1.0

    θ_y_bar = exp(s[1])
    M = s[2]
    k = exp(s[3])
    # pG = exp(s[4])
    # ζ = exp(s[5])
    pG = pG_0
    ζ = ζ_0

    damage_derivative = - d * 2.0 * 1.6 * (1.6 * M)
    return damage_derivative
end

function final_goods_production(x, s)
    e = exp(x[1])
    θ_y = exp(x[2])
    k_prime = exp(x[3])
    τ_y = x[4] / 1.0
    τ_B = x[5] / 1.0
    τ_G = x[6] / 1.0
    # τ_G = (pG_0 + x[6]) / pG_0 - 1.0

    θ_y_bar = exp(s[1])
    M = s[2]
    k = exp(s[3])
    # pG = exp(s[4])
    # ζ = exp(s[5])
    pG = pG_0
    ζ = ζ_0

    y_tilde = raw_goods_production(x, s)
    Ψ_y = goods_transition_cost(x, s)
    damage = damage_function(x, s)

    y = A * damage * Ψ_y * y_tilde

    return y

end

function energy_price_index(x, s)
    e = exp(x[1])
    θ_y = exp(x[2])
    k_prime = exp(x[3])
    τ_y = x[4] / 1.0
    τ_B = x[5] / 1.0
    τ_G = x[6] / 1.0
    # τ_G = (pG_0 + x[6]) / pG_0 - 1.0

    θ_y_bar = exp(s[1])
    M = s[2]
    k = exp(s[3])
    # pG = exp(s[4])
    # ζ = exp(s[5])
    pG = pG_0
    ζ = ζ_0

    p_e = (ω_B^σ_e * (pB + τ_B)^(1.0 - σ_e) + ω_G^σ_e * (pG * (1.0 + τ_G))^(1.0 - σ_e))^(1.0 / (1.0 - σ_e))
    return p_e
end



function total_consumption(x, s)
    e = exp(x[1])
    θ_y = exp(x[2])
    k_prime = exp(x[3])
    τ_y = x[4] / 1.0
    τ_B = x[5] / 1.0
    τ_G = x[6] / 1.0
    # τ_G = (pG_0 + x[6]) / pG_0 - 1.0

    θ_y_bar = exp(s[1])
    M = s[2]
    k = exp(s[3])
    # pG = exp(s[4])
    # ζ = exp(s[5])
    pG = pG_0
    ζ = ζ_0

    eB = eB_constraint(x, s)
    eG = eG_constraint(x, s)
    y = final_goods_production(x, s)
    i = k_prime * exp(g_A) - (1.0 - δ) * k
    c = y - eB * pB - eG * pG - i
    return c
end


function level_constraint(x, s)
    e = exp(x[1])
    θ_y = exp(x[2])
    k_prime = exp(x[3])
    τ_y = x[4] / 1.0
    τ_B = x[5] / 1.0
    τ_G = x[6] / 1.0
    # τ_G = (pG_0 + x[6]) / pG_0 - 1.0

    θ_y_bar = exp(s[1])
    M = s[2]
    k = exp(s[3])
    # pG = exp(s[4])
    # ζ = exp(s[5])
    pG = pG_0
    ζ = ζ_0

    p_e = energy_price_index(x, s)
    y_tilde = raw_goods_production(x, s)
    Ψ_y = goods_transition_cost(x, s)
    damage = damage_function(x, s)

    error = log(damage) + log(Ψ_y) + 1.0 / σ_y * log(y_tilde) +
            (-1.0 / σ_y) * ((α_y) * log(θ_y) + log(e)) + (α_y) * log(θ_y) - log(p_e)
    return error
end

function eB_constraint(x, s)
    e = exp(x[1])
    θ_y = exp(x[2])
    k_prime = exp(x[3])
    τ_y = x[4] / 1.0
    τ_B = x[5] / 1.0
    τ_G = x[6] / 1.0
    # τ_G = (pG_0 + x[6]) / pG_0 - 1.0

    θ_y_bar = exp(s[1])
    M = s[2]
    k = exp(s[3])
    # pG = exp(s[4])
    # ζ = exp(s[5])
    pG = pG_0
    ζ = ζ_0

    p_e = energy_price_index(x, s)

    eB = ((pB + τ_B) / p_e / ω_B)^(-σ_e) * e
    return eB
end

function eG_constraint(x, s)
    e = exp(x[1])
    θ_y = exp(x[2])
    k_prime = exp(x[3])
    τ_y = x[4] / 1.0
    τ_B = x[5] / 1.0
    τ_G = x[6] / 1.0
    # τ_G = (pG_0 + x[6]) / pG_0 - 1.0

    θ_y_bar = exp(s[1])
    M = s[2]
    k = exp(s[3])
    # pG = exp(s[4])
    # ζ = exp(s[5])
    pG = pG_0
    ζ = ζ_0
    p_e = energy_price_index(x, s)

    eG = ((pG * (1.0 + τ_G)) / p_e / ω_G)^(-σ_e) * e
    return eG
end

function BG_price_share(x, s)
    e = exp(x[1])
    θ_y = exp(x[2])
    k_prime = exp(x[3])
    τ_y = x[4] / 1.0
    τ_B = x[5] / 1.0
    τ_G = x[6] / 1.0
    # τ_G = (pG_0 + x[6]) / pG_0 - 1.0

    θ_y_bar = exp(s[1])
    M = s[2]
    k = exp(s[3])
    # pG = exp(s[4])
    # ζ = exp(s[5])
    pG = pG_0
    ζ = ζ_0
    p_e = energy_price_index(x, s)

    share = pG * (1.0 + τ_G) / pB
    return share
end

function y_tech_constraint(x, s)
    e = exp(x[1])
    θ_y = exp(x[2])
    k_prime = exp(x[3])
    τ_y = x[4] / 1.0
    τ_B = x[5] / 1.0
    τ_G = x[6] / 1.0
    # τ_G = (pG_0 + x[6]) / pG_0 - 1.0

    θ_y_bar = exp(s[1])
    M = s[2]
    k = exp(s[3])
    # pG = exp(s[4])
    # ζ = exp(s[5])
    pG = pG_0
    ζ = ζ_0

    y_tilde = raw_goods_production(x, s)

    γ = goods_transition_cost(x, s)
    γ_dot = derivative_goods_transition_cost(x, s)
    damage = damage_function(x, s)

    error = exp(damage) * γ_dot / θ_y_bar * y_tilde +
            exp(damage) * γ * y_tilde^(1.0 / σ_y) * ((θ_y^(α_y) * e)^(-1.0 / σ_y) * ((α_y) * θ_y^(α_y - 1.0) * e) + (θ_y^(α_y - 1.0) * k^σ)^(-1.0 / σ_y) * ((α_y - 1.0) * θ_y^(α_y - 2.0) * k^σ)) -
            τ_y

    return error
end

function y_tech_forward_constraint_test(x, s, x_prime)
    e = exp(x[1])
    θ_y = exp(x[2])
    k_prime = exp(x[3])
    τ_y = x[4] / 1.0
    τ_B = x[5] / 1.0
    τ_G = x[6] / 1.0
    # τ_G = (pG_0 + x[6]) / pG_0 - 1.0

    θ_y_bar = exp(s[1])
    M = s[2]
    k = exp(s[3])
    # pG = exp(s[4])
    # ζ = exp(s[5])
    pG = pG_0
    ζ = ζ_0

    e_prime = x_prime[1]
    θ_y_prime = exp(x_prime[2])
    k_prime_prime = exp(x[3])
    τ_y_prime = x_prime[4] / 1.0
    τ_B_prime = x_prime[5] / 10.0
    τ_G_prime = x_prime[6] / 1.0

    s_prime = law_of_motion(x, s)

    θ_y_bar_prime = exp(s_prime[1])
    M_prime = s_prime[2]
    k_prime = exp(s_prime[3])

    c = total_consumption(x, s)
    c_prime = total_consumption(x_prime, s_prime)

    y_tilde = raw_goods_production(x, s)
    y_tilde_prime = raw_goods_production(x_prime, s_prime)

    γ = goods_transition_cost(x, s)
    γ_dot = derivative_goods_transition_cost(x, s)
    γ_dot_prime = derivative_goods_transition_cost(x_prime, s_prime)
    damage = damage_function(x, s)
    damage_prime = damage_function(x_prime, s_prime)

    error = (damage * γ_dot / θ_y_bar * y_tilde +
    damage * γ * y_tilde^(1.0 / σ_y) * ((θ_y^(α_y) * e)^(-1.0 / σ_y) * ((α_y) * θ_y^(α_y - 1.0) * e) + (θ_y^(α_y - 1.0) * k^σ)^(-1.0 / σ_y) * ((α_y - 1.0) * θ_y^(α_y - 2.0) * k^σ)) -
             τ_y) +
            β * c_prime^(-α) / c^(-α) *
            (damage_prime * γ_dot_prime * (-θ_y_prime / θ_y_bar_prime^2.0) * y_tilde_prime)

    return error
end


function y_tech_forward_constraint(x, s)
    e = exp(x[1])
    θ_y = exp(x[2])
    k_prime = exp(x[3])
    τ_y = x[4] / 1.0
    τ_B = x[5] / 1.0
    τ_G = x[6] / 1.0
    # τ_G = (pG_0 + x[6]) / pG_0 - 1.0

    θ_y_bar = exp(s[1])
    M = s[2]
    k = exp(s[3])
    # pG = exp(s[4])
    # ζ = exp(s[5])
    pG = pG_0
    ζ = ζ_0

    s_prime = law_of_motion(x, s)
    e_prime = exp(e_function(s_prime))
    θ_y_prime = exp(θ_function(s_prime))

    θ_y_bar_prime = exp(s_prime[1])
    M_prime = s_prime[2]
    k_prime = exp(s_prime[3])
    x_prime = [e_function(s_prime), θ_function(s_prime), 0.0, 0.0, 0.0, 0.0]

    c = total_consumption(x, s)
    c_prime = c_function(s_prime)

    y_tilde = raw_goods_production(x, s)
    y_tilde_prime = raw_goods_production(x_prime, s_prime)
    γ = goods_transition_cost(x, s)
    γ_dot = derivative_goods_transition_cost(x, s)
    γ_dot_prime = derivative_goods_transition_cost(x_prime, s_prime)
    damage = damage_function(x, s)
    damage_prime = damage_function(x_prime, s_prime)

    error = (damage * γ_dot / θ_y_bar * y_tilde +
             damage * γ * y_tilde^(1.0 / σ_y) * ((θ_y^(α_y) * e)^(-1.0 / σ_y) * ((α_y) * θ_y^(α_y - 1.0) * e) + (θ_y^(α_y - 1.0) * k^σ)^(-1.0 / σ_y) * ((α_y - 1.0) * θ_y^(α_y - 2.0) * k^σ)) -
             τ_y) +
            β * c_prime^(-α) / c^(-α) *
            (damage_prime * γ_dot_prime * (-θ_y_prime / θ_y_bar_prime^2.0) * y_tilde_prime)


    return error
end

function capital_euler_constraint_test(x, s, x_prime)
    e = exp(x[1])
    θ_y = exp(x[2])
    k_prime = exp(x[3])
    τ_y = x[4] / 1.0
    τ_B = x[5] / 1.0
    τ_G = x[6] / 1.0
    # τ_G = (pG_0 + x[6]) / pG_0 - 1.0

    θ_y_bar = exp(s[1])
    M = s[2]
    k = exp(s[3])
    # pG = exp(s[4])
    # ζ = exp(s[5])
    pG = pG_0
    ζ = ζ_0

    s_prime = law_of_motion(x, s)

    θ_y_prime = exp(x_prime[2])

    θ_y_bar_prime = exp(s_prime[1])
    M_prime = s_prime[2]
    k_prime = exp(s_prime[3])

    c = total_consumption(x, s)
    c_prime = total_consumption(x_prime, s_prime)

    γ_prime = goods_transition_cost(x_prime, s_prime)
    damage = damage_function(x, s)
    damage_prime = damage_function(x_prime, s_prime)


    error = -c^(-α) * exp(g_A) +
            β * c_prime^(-α) *
            (damage_prime * γ_prime * dytildedk(x_prime, s_prime) + 1.0 - δ)
    return error
end


function dytildedk(x, s)
    e = exp(x[1])
    θ_y = exp(x[2])
    k_prime = exp(x[3])
    τ_y = x[4] / 1.0
    τ_B = x[5] / 1.0
    τ_G = x[6] / 1.0
    # τ_G = (pG_0 + x[6]) / pG_0 - 1.0

    θ_y_bar = exp(s[1])
    M = s[2]
    k = exp(s[3])
    # pG = exp(s[4])
    # ζ = exp(s[5])
    pG = pG_0
    ζ = ζ_0

    y_tilde = raw_goods_production(x, s)
    dyoverdk = y_tilde^(1.0 / σ_y) *
               (θ_y^(α_y - 1.0) * k^σ)^(-1.0 / σ_y) *
               θ_y^(α_y - 1.0) * σ * k^(σ - 1.0)
    return dyoverdk
end

function capital_euler_constraint(x, s)
    e = exp(x[1])
    θ_y = exp(x[2])
    k_prime = exp(x[3])
    τ_y = x[4] / 1.0
    τ_B = x[5] / 1.0
    τ_G = x[6] / 1.0
    # τ_G = (pG_0 + x[6]) / pG_0 - 1.0

    θ_y_bar = exp(s[1])
    M = s[2]
    k = exp(s[3])
    # pG = exp(s[4])
    # ζ = exp(s[5])
    pG = pG_0
    ζ = ζ_0

    s_prime = law_of_motion(x, s)
    x_prime = [e_function(s_prime), θ_function(s_prime), 0.0, 0.0, 0.0, 0.0]
    θ_y_prime = exp(x_prime[2])

    θ_y_bar_prime = exp(s_prime[1])
    M_prime = s_prime[2]
    k_prime = exp(s_prime[3])

    c = total_consumption(x, s)
    c_prime = c_function(s_prime)

    γ_prime = goods_transition_cost(x_prime, s_prime)

    damage = damage_function(x, s)
    damage_prime = damage_function(x_prime, s_prime)

    error = -c^(-α) * exp(g_A) +
            β * c_prime^(-α) *
            (damage_prime * γ_prime * dytildedk(x_prime, s_prime) + 1.0 - δ)
    return error
end





function law_of_motion(x, s)
    e = exp(x[1])
    θ_y = exp(x[2])
    k_prime = exp(x[3])
    τ_y = x[4] / 1.0
    τ_B = x[5] / 1.0
    τ_G = x[6] / 1.0
    # τ_G = (pG_0 + x[6]) / pG_0 - 1.0

    θ_y_bar = exp(s[1])
    M = s[2]
    k = exp(s[3])
    # pG = exp(s[4])
    # ζ = exp(s[5])
    pG = pG_0
    ζ = ζ_0

    eB = eB_constraint(x, s)
    M_prime = δ_M * M + ζ * eB
    # pG_prime = log(pG * (1.0 + pG_g))
    # ζ_prime = log(ζ) + ζ_g

    # return [x[2], M_prime, x[3], pG_prime, ζ_prime]
    return [x[2], M_prime, x[3]]
end

function fUtility_flow(c)
    if c <= 1e-8
        u = -1.0e10
    else
        if α == 1.0
            u = log(c)
        else
            u = c^(1.0 - α) / (1.0 - α)
        end
    end
    return u
end



function fValue_today(x, s)
    # Notice: for optim, it is essential for unpack the control variable
    e = exp(x[1])
    θ_y = exp(x[2])
    k_prime = exp(x[3])
    τ_y = x[4] / 1.0
    τ_B = x[5] / 1.0
    τ_G = x[6] / 1.0
    # τ_G = (pG_0 + x[6]) / pG_0 - 1.0

    θ_y_bar = exp(s[1])
    M = s[2]
    k = exp(s[3])
    # pG = exp(s[4])
    # ζ = exp(s[5])
    pG = pG_0
    ζ = ζ_0

    β_effective = β

    s_tomorrow = law_of_motion(x, s)
    c = total_consumption(x, s)
    flow = fUtility_flow(c)
    EV = fValue_function(s_tomorrow)
    V = flow + β_effective * EV
    return V
end

function value_function_iteration(x0, s_today, control_LB_today = control_LB, control_UB_today = control_UB)

    total_consumption_handle(x) = total_consumption(x, s_today)
    fValue_today_handle(x) = fValue_today(x, s_today)


    level_constraint_handle(x) = level_constraint(x, s_today)
    y_tech_constraint_handle(x) = y_tech_forward_constraint(x, s_today)
    capital_euler_constraint_handle(x) = capital_euler_constraint(x, s_today)


    function myfunc(x::Vector, grad::Vector)
        fun(x) = -1.0 * (fValue_today_handle(x))
        if length(grad) > 0
            grad_temp = ForwardDiff.gradient(fun, x)
            grad[:] = grad_temp[:]
        end
        return fun(x)
    end

    function myfunc_init(x::Vector, grad::Vector)
        fun(x) = -1.0 * (total_consumption_handle(x))
        if length(grad) > 0
            grad_temp = ForwardDiff.gradient(fun, x)
            grad[:] = grad_temp[:]
        end
        return fun(x)
    end

    function myconstraint1(x::Vector, grad::Vector)
        if length(grad) > 0
            grad_temp = ForwardDiff.gradient(level_constraint_handle, x)
            grad[:] = grad_temp[:]
        end
        level_constraint_handle(x)
    end

    function myconstraint2(x::Vector, grad::Vector)
        if length(grad) > 0
            grad_temp = ForwardDiff.gradient(y_tech_constraint_handle, x)
            grad[:] = grad_temp[:]
        end
        y_tech_constraint_handle(x)
    end

    function myconstraint3(x::Vector, grad::Vector)
        if length(grad) > 0
            grad_temp = ForwardDiff.gradient(capital_euler_constraint_handle, x)
            grad[:] = grad_temp[:]
        end
        capital_euler_constraint_handle(x)
    end


    opt = Opt(:LD_SLSQP, 6)
    opt.xtol_rel = 1e-10
    opt.maxeval = 10000
    opt.min_objective = myfunc
    equality_constraint!(opt, (x, g) -> myconstraint1(x, g), 1e-6)
    equality_constraint!(opt, (x, g) -> myconstraint2(x, g), 1e-6)
    equality_constraint!(opt, (x, g) -> myconstraint3(x, g), 1e-6)
    opt.lower_bounds = control_LB_today
    opt.upper_bounds = control_UB_today
    (minf, minx, ret) = NLopt.optimize(opt, x0)

    t_value = minf
    optimal_control = minx

    return [t_value, optimal_control, string(ret)]
end

function value_function_iteration_initial(x0, s_today)

    total_consumption_handle(x) = total_consumption(x, s_today)
    fValue_today_handle(x) = fValue_today(x, s_today)


    level_constraint_handle(x) = level_constraint(x, s_today)
    y_tech_constraint_handle(x) = y_tech_forward_constraint(x, s_today)
    capital_euler_constraint_handle(x) = capital_euler_constraint(x, s_today)


    function myfunc(x::Vector, grad::Vector)
        fun(x) = -1.0 * (fValue_today_handle(x))
        if length(grad) > 0
            grad_temp = ForwardDiff.gradient(fun, x)
            grad[:] = grad_temp[:]
        end
        return fun(x)
    end

    function myfunc_init(x::Vector, grad::Vector)
        fun(x) = -1.0 * (total_consumption_handle(x))
        if length(grad) > 0
            grad_temp = ForwardDiff.gradient(fun, x)
            grad[:] = grad_temp[:]
        end
        return fun(x)
    end

    function myconstraint1(x::Vector, grad::Vector)
        if length(grad) > 0
            grad_temp = ForwardDiff.gradient(level_constraint_handle, x)
            grad[:] = grad_temp[:]
        end
        level_constraint_handle(x)
    end

    function myconstraint2(x::Vector, grad::Vector)
        if length(grad) > 0
            grad_temp = ForwardDiff.gradient(y_tech_constraint_handle, x)
            grad[:] = grad_temp[:]
        end
        y_tech_constraint_handle(x)
    end

    function myconstraint3(x::Vector, grad::Vector)
        if length(grad) > 0
            grad_temp = ForwardDiff.gradient(capital_euler_constraint_handle, x)
            grad[:] = grad_temp[:]
        end
        capital_euler_constraint_handle(x)
    end


    opt = Opt(:LD_SLSQP, 6)
    opt.xtol_rel = 1e-10
    opt.maxeval = 10000
    opt.min_objective = myfunc
    equality_constraint!(opt, (x, g) -> myconstraint1(x, g), 1e-6)
    opt.lower_bounds = control_LB
    opt.upper_bounds = control_UB
    (minf, minx, ret) = NLopt.optimize(opt, x0)

    t_value = minf
    optimal_control = minx

    return [t_value, optimal_control, string(ret)]
end


function value_function_iteration_no_constraint(x0, s_today)

    total_consumption_handle(x) = total_consumption(x, s_today)
    fValue_today_handle(x) = fValue_today(x, s_today)

    function myfunc(x::Vector, grad::Vector)
        fun(x) = -1.0 * (fValue_today_handle(x))
        if length(grad) > 0
            grad_temp = ForwardDiff.gradient(fun, x)
            grad[:] = grad_temp[:]
        end
        return fun(x)
    end


    opt = Opt(:LD_LBFGS, 6)
    opt.xtol_rel = 1e-10
    opt.maxeval = 10000
    opt.min_objective = myfunc
    opt.lower_bounds = control_LB
    opt.upper_bounds = control_UB
    (minf, minx, ret) = NLopt.optimize(opt, x0)

    t_value = minf
    optimal_control = minx

    return [t_value, optimal_control, string(ret)]
end