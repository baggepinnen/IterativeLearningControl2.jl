using IterativeLearningControl
using ControlSystemsBase, LinearAlgebra
using Plots
using Test


function double_mass_model(; 
    Jm = 1,   # motor inertia
    Jl = 1,   # load inertia
    k  = 100, # stiffness
    c0 = 1,   # motor damping
    c1 = 1,   # transmission damping
    c2 = 1,   # load damping
)

    A = [
        0.0 1 0 0
        -k/Jm -(c1 + c0)/Jm k/Jm c1/Jm
        0 0 0 1
        k/Jl c1/Jl -k/Jl -(c1 + c2)/Jl
    ]
    B = [0, 1/Jm, 0, 0]
    C = [1 0 0 0]
    ss(A, B, C, 0)
end

function funnysin(x)
    x = sin(x)
    s,a = sign(x), abs(x)
    s*((a + 0.01)^0.2 - 0.01^0.2)
end

@testset "IterativeLearningControl.jl" begin

    # Continuous
    G    = double_mass_model(Jl = 1)
    Gact = double_mass_model(Jl = 1.5) # 50% more load than modeled
    C  = pid(10, 1, 1, form = :series) * tf(1, [0.02, 1])
    Ts = 0.02 # Sample time
    z = tf("z", Ts)


    # Discrete
    Gr = c2d(feedback(G*C), Ts)       |> tf
    Gu = c2d(feedback(G, C), Ts)
    Gract = c2d(feedback(Gact*C), Ts)
    Guact = c2d(feedback(Gact, C), Ts)

    sim = (r, a) -> lsim([Gr Gu], [r; a], t)
    simact = (r, a) -> lsim([Gract Guact], [r; a], t)
    T = 3pi    # Duration
    t = 0:Ts:T # Time vector


    r = funnysin.(t)' |> Array # Reference signal
    N = length(r)
    
    Q = c2d(tf(1, [0.05, 1]), Ts)
    # L = 0.9z^1 # A more conservative and heuristic choice
    L = 0.5inv(Gr) # Make the scaling factor smaller to take smaller steps

    prob = ILCProblem(r, simact)
    alg1 = HeuristicILC(Q, L, t)
    alg2 = OptimizationILC(Gu; N, ρ=0.00001, λ=0.0001)
    sol1 = ilc(prob, alg1)
    sol2 = ilc(prob, alg2)

    @test all(diff(norm.(sol1.E)) .< 0)
    @test all(diff(norm.(sol2.E)) .< 0)
    @test all(norm.(sol2.E) .<= norm.(sol2.E))

    plot(sol1)
end
