using IterativeLearningControl
using ControlSystemsBase, LinearAlgebra
using Plots
using Test

using JuMP, BlockArrays, OSQP

using IterativeLearningControl: hv, hankel_operator, mv_hankel_operator
import IterativeLearningControl as ILC

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

function centraldiff(v::AbstractMatrix)
    dv = Base.diff(v, dims=1)/2
    a1 = [dv[[1],:];dv]
    a2 = [dv;dv[[end],:]]
    a1+a2
end

function centraldiff(v::AbstractVector)
    dv = Base.diff(v)/2
    a1 = [dv[1];dv]
    a2 = [dv;dv[end]]
    a1+a2
end

@testset "IterativeLearningControl.jl" begin


    @testset "hankel operator" begin
        @info "Testing hankel operator"
        G = ssrand(1,1,5,Ts=1)
        N = 30
        u = randn(1, N)
        OP = hankel_operator(G, N)
        y = lsim(G, u).y
        y2 = (OP*u')'
        @test y ≈ y2

        OP2 = hankel_operator(LTVSystem(fill(G, N)))
        @test OP ≈ OP2
    end

    @testset "mv hankel operator" begin
        @info "Testing mv hankel operator"
        G = ssrand(2,3,5,Ts=1)
        N = 12
        u = randn(G.nu, N)
        OP = mv_hankel_operator(G, N)
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
    
    A = function (model, v)
        l,u = (-25ones(Gu.nu), 25ones(Gu.nu))
        JuMP.@constraint(model, [i=1:size(v, 2)], l .<= v[:, i] .<= u)
    end
    
    Y = function (model, yh)
        l,u = -1.1ones(Gr.ny), 1.1ones(Gr.ny)
        JuMP.@constraint(model, [i=1:size(yh, 2)], l .<= yh[:, i] .<= u)
    end
    
    alg2 = ConstrainedILC(; Q, R, A, Y, opt=OSQP.Optimizer, verbose=false, α=1)
    sol2 = ilc(prob, alg2)
    plot(sol2)
    ## Look at the total plant input
    Gr2 = [Gr; c2d(feedback(C, G), Ts)]
    Gu2 = [Gu; c2d(feedback(C*G), Ts)]
    plot(lsim([Gr2 Gu2], [r; sol2.A[end]]))
    
    @test all(diff(norm.(sol1.E)) .< 0)
    @test all(diff(norm.(sol2.E)) .< 0)

    @test norm(sol2.E[end]) ≈ 4.4544077312015835 atol = 1e-1


@testset "multivariate" begin
    @info "Testing multivariate"
    # Gr2 = append(Gr, c2d(feedback(C, G), Ts))
    Gr_constraints = [Gr; c2d(feedback(C, G), Ts)]
    Gu_constraints = [Gu; c2d(feedback(C*G), Ts)]

    Q = 1000I(Gr.ny)
    R = 0.001I(Gu.nu)
    
    A = function (model, v)
        l,u = (-25ones(Gu_constraints.nu), 25ones(Gu_constraints.nu))
        JuMP.@constraint(model, [i=1:size(v, 2)], l .<= v[:, i] .<= u)
    end
    
    Y = function (model, yh)
        u = [1.1, 1000]
        l = -u
        JuMP.@constraint(model, [i=1:size(yh, 2)], l .<= yh[:, i] .<= u)
    end
    
    # r2 = [r; zero(r)]
    # prob = ILCProblem(; r=r2, Gr=Gr2, Gu=Gu2)
    prob = ILCProblem(; r, Gr, Gu)
    alg = ConstrainedILC(; Gr_constraints, Gu_constraints, Q, R, A, Y, opt=OSQP.Optimizer, verbose=false, α=1)
    workspace = init(prob, alg);
    sol = ilc(prob, alg; iters=5)
    plot(sol)


    constrained_res = lsim([Gr_constraints Gu_constraints], [r; sol.A[end]])
    plot(constrained_res)
    @test all(-1000.01 .<= constrained_res.y[2,:] .<= 1000.01)
    ##
    
    @test all(diff(norm.(sol.E)) .< 0)

    @test norm(sol.E[end]) ≈ 4.940690279813115 atol = 1e-1
end


@testset "ModelFreeILC" begin
    @info "Testing ModelFreeILC"
    r = (funnysin.(t)') |> Array
    prob = ILCProblem(; r, Gr, Gu)
    actual = ILCProblem(; r, Gr=Gract, Gu=Guact)
    alg = ModelFreeILC(1, 500)
    sol = ilc(prob, alg; actual, iters=10)
    plot(sol)
    @test all(diff(norm.(sol.E)) .< 0)
    @test norm(sol.E[end]) ≈ 0.9347317369952786 atol = 1e-2
    
    sol2 = ILC.ilc(prob, alg; actual, iters=10, store_all=true)
    @test length(sol2.Y) == 3length(sol.Y)
    @test norm(sol2.E[end]) ≈ 0.9347317369952786 atol = 1e-2
end

@testset "GradientILC" begin
    @info "Testing GradientILC"
    r = (funnysin.(t)') |> Array
    prob = ILCProblem(; r, Gr, Gu)
    actual = ILCProblem(; r, Gr=Gract, Gu=Guact)
    alg = ILC.GradientILC(500)
    sol = ilc(prob, alg; actual, iters=10)
    plot(sol)
    @test all(diff(norm.(sol.E)) .< 0)
    @test norm(sol.E[end]) ≈ 0.8851911872861579 atol = 1e-2

end




##


@testset "LTVSystem and MIMO" begin
    @info "Testing LTVSystem and MIMO"

    @testset "hankel" begin
        N = 30
        u = randn(2, N)
        syss = LTVSystem([ssrand(2,2,2,Ts=3) for i = 1:N])
        @test syss.nu == 2
        @test syss.ny == 2
        @test syss.nx == 2
        @test syss.N == N
        @test syss.Ts == 3
        OP = hankel_operator(syss)
        res = lsim(syss, u)
        y1 = res.y
        res2 = lsim(syss, u)
        Y2 = res2.y
        Y3 = reshape((OP*hv(u)), :, syss.ny)'
        @test Y2 ≈ Y3
    end


    # Continuous
    P1    = DemoSystems.double_mass_model(Jl = 1, outputs=1)
    P    = DemoSystems.double_mass_model(Jl = 1, outputs=1:2)
    Pact = DemoSystems.double_mass_model(Jl = 1.5, outputs=1:2) # 50% more load than modeled

    
    C  = [pid(10, 1, 1, form = :series) * tf(1, [0.02, 1]) 0]
    
    
    # Discrete
    Ts = 0.02 # Sample time
    Pd = c2d(P, Ts)
    Pd1 = c2d(P1, Ts)
    Pdact = c2d(Pact, Ts)
    Cd = c2d(ss(C), Ts, :tustin)
    Gr = c2d(feedback(P*C), Ts)
    Gu = c2d(feedback(P, C), Ts)
    Gract = c2d(feedback(Pact*C), Ts)
    Guact = c2d(feedback(Pact, C), Ts)


    P2 = LTVSystem([Pd, Pd])
    P12 = LTVSystem([Pd1, Pd1])
    @test P2*Cd ≈ LTVSystem(fill(Pd*Cd, 2))
    @test feedback(P2*Cd) ≈ LTVSystem(fill(feedback(Pd*Cd), 2))
    @test feedback(P2, Cd) ≈ LTVSystem(fill(feedback(Pd, Cd), 2))
    @test feedback(Cd, P2) ≈ LTVSystem(fill(feedback(Cd, Pd), 2))
    @test feedback(P12, P12) ≈ LTVSystem(fill(feedback(Pd1, Pd1), 2))


    
    T = 3pi    # Duration
    t = 0:Ts:T # Time vector
    r = funnysin.(t)' |> Array # Reference signal
    r = [r; centraldiff(r')' ./ Ts]

    prob = ILCProblem(; r, Gr, Gu)
    alg = OptimizationILC(; ρ=0.00001, λ=0.0001)
    sol = ilc(prob, alg; iters=5)
    plot(sol)
    @test all(diff(norm.(sol.E)) .< 0)
    @test norm(sol.E[end]) ≈ 4.906366002179874 atol = 1e-2

    alg = ConstrainedILC(; Q=1000I(Gr.ny), R=0.001I(Gu.nu), opt=OSQP.Optimizer, verbose=false, α=1)
    sol = ilc(prob, alg; iters=5)
    plot(sol)
    @test all(diff(norm.(sol.E)) .< 0)
    @test norm(sol.E[end]) ≈ 4.956076021566923 atol = 1e-2

    alg = ModelFreeILC(1, 1)
    probr = ILCProblem(; r, Gr, Gu=Gr)
    sol = ilc(probr, alg; iters=10)
    plot(sol)
    @test all(diff(norm.(sol.E)) .< 0)
    @test norm(sol.E[end]) ≈ 7.7459826918966925 atol = 1e-2

    alg = GradientILC(1e-2)
    sol = ilc(probr, alg; iters=10)
    plot(sol)
    @test all(diff(norm.(sol.E)) .< 0)
    @test norm(sol.E[end]) ≈ 6.181388606199837 atol = 1e-2



    # LTV ILC tests
    N = size(r, 2)
    r = [r r]
    Gr = convert(LTVSystem{Float64, N}, Gr)
    Gract = convert(LTVSystem{Float64, N}, Gract)
    Gr = tcat(Gr, Gract)

    Gu = convert(LTVSystem{Float64, N}, Gu)
    Guact = convert(LTVSystem{Float64, N}, Guact)
    Gu = tcat(Gu, Guact)

    GrGu = [Gr Gu]


    prob = ILCProblem(; r, Gr, Gu)
    alg = OptimizationILC(; ρ=0.00001, λ=0.0001)
    sol = ilc(prob, alg; iters=5)
    plot(sol)
    @test all(diff(norm.(sol.E)) .< 0)
    @test norm(sol.E[end]) ≈ 4.933120956585128 atol = 1e-2

    alg = ConstrainedILC(; Q=1000I(Gr.ny), R=0.001I(Gu.nu), opt=OSQP.Optimizer, verbose=false, α=1)
    sol = ilc(prob, alg; iters=5)
    plot(sol)
    @test all(diff(norm.(sol.E)) .< 0)
    @test norm(sol.E[end]) ≈ 5.049410033148527 atol = 1e-2

    alg = ModelFreeILC(1, 1)
    probr = ILCProblem(; r, Gr, Gu=Gr)
    sol = ilc(probr, alg; iters=10)
    plot(sol)
    @test all(diff(norm.(sol.E)) .< 0)
    @test norm(sol.E[end]) ≈ 11.162781721731639 atol = 1e-2

    alg = GradientILC(1e-2)
    sol = ilc(probr, alg; iters=10)
    plot(sol)
    @test all(diff(norm.(sol.E)) .< 0)
    @test norm(sol.E[end]) ≈ 8.158987305624235 atol = 1e-2

    f = function (x, a, r, p, t)
        ra = [r; a]
        i = round(Int, t / Ts + 1)
        GrGu.A[:,:,i]*x + GrGu.B[:,:,i]*ra
    end
    g = function (x, a, r, p, t)
        ra = [r; a]
        i = round(Int, t / Ts + 1)
        GrGu.C[:,:,i]*x + GrGu.D[:,:,i]*ra
    end
    model = NonlinearSystem(; f, g, GrGu.nx, GrGu.ny, Ts, na=Gu.nu)

    prob = ILC.NonlinearILCProblem(; r, model, x0=zeros(model.nx))
    alg = OptimizationILC(; ρ=0.00001, λ=0.0001)
    sol = ilc(prob, alg; iters=5)
    plot(sol)
    @test all(diff(norm.(sol.E)) .< 0)
    @test norm(sol.E[end]) ≈ 4.933120956585128 atol = 1e-2


    f = @views function (x, a, r, p, t)
        ra = r+a
        i = round(Int, t / Ts + 1)
        Gr.A[:,:,i]*x + Gr.B[:,:,i]*ra
    end
    g = @views function (x, a, r, p, t)
        ra = r+a
        i = round(Int, t / Ts + 1)
        Gr.C[:,:,i]*x + Gr.D[:,:,i]*ra
    end
    modelr = NonlinearSystem(; f, g, Gr.nx, Gr.ny, Ts, na=Gr.nu)

    probr = ILC.NonlinearILCProblem(; r, model=modelr, x0=zeros(modelr.nx))

    alg = ModelFreeILC(1, 1)
    sol = ilc(probr, alg; iters=10)
    plot(sol)
    @test all(diff(norm.(sol.E)) .< 0)
    @test norm(sol.E[end]) ≈ 11.162781721731639 atol = 1e-2

    alg = GradientILC(1e-2)
    probr = ILC.NonlinearILCProblem(; r, model=modelr, x0=zeros(modelr.nx))
    sol = ilc(probr, alg; iters=10)
    plot(sol)
    @test all(diff(norm.(sol.E)) .< 0)
    @test norm(sol.E[end]) ≈ 8.158987305624235 atol = 1e-2



    A = function (model, v)
        l,u = (-35ones(1), 35ones(1))
        JuMP.@constraint(model, [i=1:size(v, 2)], l .<= v[:, i] .<= u)
    end
    
    Y = function (model, yh)
        u = [1.1, 5]
        l = -u
        JuMP.@constraint(model, [i=1:size(yh, 2)], l .<= yh[:, i] .<= u)
    end
    alg = ConstrainedILC(; Q=1000I(model.ny), R=0.001I(model.na), A, Y, opt=OSQP.Optimizer, verbose=false, α=1)
    sol = ilc(prob, alg; iters=5)
    plot(sol)
    @test all(diff(norm.(sol.E)) .< 0)
    @test norm(sol.E[end]) ≈ 10.337380903140984 atol = 1e-2
    @test all(-1.11 .<= sol.Y[end][1,:] .<= 1.11)
    @test all(-35.02 .<= sol.A[end] .<= 35.02)
end
end
