const A = 1.0
const g_A = 0.023
# const g_A = 0.0

const α = 1.45

const β = 0.985 * exp((1.0 - α) * g_A)
const T = 300


const α_y = 1.0 - 0.031 # LR Share of e in y is 1.0 - α_y
# const σ_y = 0.05 # SR elasticity between k and e
const σ_y = 0.25 # SR elasticity between k and e
const ρ_y = (σ_y - 1.0) / σ_y
# const γ_y = 0.99 # transition cost parameter
# const γ_y = 5000.0 # transition cost paramete
# const γ_y = 10.0
# const γ_y = 4.9
const γ_y = 2.498
const ζ_g = 0.0

const d = 0.003467
# const d = -0.0
# const d = 0.01 # Howard and Sterner 2017 Damage 

const σ = 0.35 / α_y

const δ_M = 1.00

const δ = 0.06

const k_ss = exp(1.71)
# const ρ_e = -0.058
const ρ_e = 0.44
const σ_e = 1.0 / (1.0 - ρ_e)
const ω_G = 0.356
const ω_B = 1.0 - ω_G
const pB = 74.0 / 1000.0 
const pG_0 = 600.0 / 1000.0 
const pG_g = -0.0

# const θ_y_ss = 2.57489 # (ρ_e = -0.058)
# const θ_y_ss = 2.033 # (ρ_e = 0.44)
const θ_y_ss = 3.04021 # (σ_y = 0.25)


# const ζ_capita = 3.8 # (ρ_e = -0.058)
const ζ_capita = 2.0429731070841366 # (ρ_e = 0.44)



# Bounds
const k_LB = log(0.3 * k_ss)
const k_UB = log(1.2 * k_ss)

const θ_y_LB = log(exp(θ_y_ss) * 0.5)
const θ_y_UB = log(exp(θ_y_ss) * 5.0)

const M_LB = 0.0
const M_UB = 5.0

const pG_LB = log(pG_0 / 10.0)
const pG_UB = log(pG_0 * 10.0)


const ζ_0 = ζ_capita * 7.5 / 1000.0
const ζ_LB = log(ζ_0 * 0.2)
const ζ_UB = log(ζ_0 * 5.0)

# const LB = [θ_y_LB, M_LB, k_LB, pG_LB, ζ_LB]
# const UB = [θ_y_UB, M_UB, k_UB, pG_UB, ζ_UB]

const LB = [θ_y_LB, M_LB, k_LB]
const UB = [θ_y_UB, M_UB, k_UB]


const D = length(LB)


const e_LB = log(1e-2)
const e_UB = log(500.0)
