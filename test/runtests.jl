using IterativeLearningControl
using ControlSystemsBase, LinearAlgebra
using Plots
using Test

using JuMP, BlockArrays, Ipopt

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


    @testset "hankel operator" begin
        @info "Testing hankel operator"
        G = ssrand(1,1,5,Ts=1)
        N = 30
        u = randn(1, N)
        OP = IterativeLearningControl.hankel_operator(G, N)
        y = lsim(G, u).y
        y2 = (OP*u')'
        @test y ≈ y2
    end

    @testset "mv hankel operator" begin
        @info "Testing mv hankel operator"
        G = ssrand(2,3,5,Ts=1)
        N = 12
        u = randn(G.nu, N)
        OP = IterativeLearningControl.mv_hankel_operator(G, N)
        y = lsim(G, u).y
        y2 = reshape((OP*vec(u')), :, G.ny)'
        @test y ≈ y2
    end

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

    T = 3pi    # Duration
    t = 0:Ts:T # Time vector

    r = funnysin.(t)' |> Array # Reference signal
    N = length(r)
    
    Q1 = c2d(tf(1, [0.05, 1]), Ts)
    Q3 = c2d(tf(1, [0.1, 1]), Ts)
    # L = 0.9z^1 # A more conservative and heuristic choice
    L1 = 0.5inv(Gr) # Make the scaling factor smaller to take smaller steps
    L3 = 0.5inv(tf(Gu))

    prob = ILCProblem(; r, Gr, Gu)
    actual = ILCProblem(; r, Gr=Gract, Gu=Guact)
    alg1 = HeuristicILC(Q1, L1, :ref)
    alg2 = OptimizationILC(; ρ=0.00001, λ=0.0001)
    alg3 = HeuristicILC(Q3, L3, :input)
    sol1 = ilc(prob, alg1; actual)
    sol2 = ilc(prob, alg2; actual)
    sol3 = ilc(prob, alg3; actual)

    @test all(diff(norm.(sol1.E)) .< 0)
    @test all(diff(norm.(sol2.E)) .< 0)
    @test all(diff(norm.(sol3.E)) .< 0)
    @test all(norm.(sol2.E) .<= norm.(sol2.E))

    @test norm(sol1.E[end]) ≈ 0.6358364794186305 atol = 1e-2
    @test norm(sol2.E[end]) ≈ 0.5615131021547797 atol = 1e-2
    @test norm(sol3.E[end]) ≈ 1.416392979780404 atol = 1e-2

    plot(sol1)

    ilc_theorem(alg1, Gr, tf(Gract))


    ## Test ConstrainedILC with hard step reference
    Q1 = c2d(tf(1, [0.2, 1]), Ts)
    L1 = 0.7inv(tf(Gu))
    r = sign.(funnysin.(t)')
    prob = ILCProblem(; r, Gr, Gu)
    actual = ILCProblem(; r, Gr=Gract, Gu=Guact)
    alg1 = HeuristicILC(Q1, L1, :input)
    sol1 = ilc(prob, alg1; actual)
    plot(sol1)
    
    ##
    
    
    Q = 1000I(Gr.ny)
    R = 0.001I(Gu.nu)
    
    U = function (model, v)
        l,u = (-25ones(Gu.nu), 25ones(Gu.nu))
        JuMP.@constraint(model, [i=1:size(v, 2)], l .<= v[:, i] .<= u)
    end
    
    Y = function (model, yh)
        l,u = -1.1ones(Gr.ny), 1.1ones(Gr.ny)
        JuMP.@constraint(model, [i=1:size(yh, 2)], l .<= yh[:, i] .<= u)
    end
    
    alg2 = ConstrainedILC(; Q, R, U, Y, opt=Ipopt.Optimizer, verbose=true, α=1)
    sol2 = ilc(prob, alg2)
    plot(sol2)
    ##
    
    @test all(diff(norm.(sol1.E)) .< 0)
    @test all(diff(norm.(sol2.E)) .< 0)

    @test norm(sol2.E[end]) ≈ 4.4544077312015835 atol = 1e-1
end



