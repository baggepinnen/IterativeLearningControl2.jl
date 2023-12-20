# Heuristic ILC

A heuristic ILC scheme that operates by adjusting the reference signal ``r`` typically looks something like this, at ILC iteration $k$:
```math
\begin{aligned}
y_k(t) &= G_r(q) \big(r(t) + a_k(t) \big) \\
e_k(t) &= r(t) - y_k(t) \\
a_k(t) &= Q(q) \big( a_{k-1}(t) + L(q) e_{k-1}(t) \big)
\end{aligned}
```
where $q$ is the time-shift operator, $G_r(q)$ is the transfer function from the reference $r$ to the output $y$, i.e, typically a closed-loop transfer function, $e_k$ is the control error and $a_k$ is the ILC adjustment signal, an additive correction to the reference that is learned throughout the ILC iterations in order to minimize the control error. $Q(q)$ and $L(q)$ are stable filters that control the learning dynamics. Interestingly, these filters does not have to be causal since they operate on the signals $e$ and $a$ *between* ILC iterations, when the whole signals are available at once for acausal filtering. 

If the ILC instead operates by adding directly to the the plant input ``u``, the first equation above is replaced by
```math
y_k(t) = G_r(q) r(t) + G_u(q) a_k(t)
```
where the transfer function $G_u(q)$ is the closed-loop transfer function from plant input to the output $y$.

In simulation (the rollout $y_k = G_r(q) (r + a_k)$ is simulated), this scheme is nothing other than an open-loop optimal-control strategy, while if $y_k = G_r(q) (r + a_k)$ amounts to performing an actual experiment on a process, ILC turns into episode-based reinforcement learning or adaptive control.

The system to control in this example is a double-mass system with a spring and damper in between. This system is a common model of a servo system where one mass represents the motor and the other represents the load. The spring and damper represents a flexible transmission between them. We will create two instances of the system model. ``G`` represents the nominal model, whereas ``G_{act}`` represents the actual (unknown) dynamics. This simulates a model-based approach where there is a slight error in the model. The error will lie in the mass of the load, simulating, e.g., that the motor is driving a heavier load than specified. 

# Example

## System model and controller

```@example HEURISTIC_ILC
using Plots
default(size=(800,800))
```

```@example HEURISTIC_ILC
using IterativeLearningControl, ControlSystemsBase, Plots

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

bodeplot([Gr, Gract], lab=["G model" "G actual"], plotphase=false)
```
We will design a PID controller with a filter for the system, the controller is poorly tuned and not very good at tracking fast reference steps, in practice, one would likely design a feedforward controller as well to improve upon this, but for now we'll stick with the simple feedback controller.

```@example HEURISTIC_ILC
C  = pid(10, 1, 1, form = :series) * tf(1, [0.02, 1])
Ts = 0.02 # Sample time
plot(step(Gr, 10), title="Closed-loop step response", lab="model")
plot!(step(Gract, 10), lab="actual")
```

## Reference trajectory

Next up we design a reference trajectory and simulate the actual closed-loop dynamics.
```@example HEURISTIC_ILC
T = 3pi    # Duration
t = 0:Ts:T # Time vector
function funnysin(x)
    x = sin(x)
    s,a = sign(x), abs(x)
    s*((a + 0.01)^0.2 - 0.01^0.2)
end
r = funnysin.(t)' |> Array # Reference signal

res = lsim(Gract, r, t)
plot(res, plotu=true, layout=1, sp=1, title="Closed-loop simulation with actual dynamics", lab=["y" "r"])
```
Performance is poor.. Enter ILC!

## Choosing filters
The next step is to define the ILC filters ``Q(z)`` and ``L(z)``.

The filter $L(q)$ acts as a frequency-dependent step size. To make the procedure take smaller steps, simply scale $L$ by a constant < 1. Scaling down $L$ makes the learning process slower but more robust. A heuristic choice of $L$ is some form of scaled lookahead, such as $0.5z^l$ where $l \geq 0$ is the number of samples lookahead. A model-based approach may use some form of inverse of the system model, which is what we will use here. [^nonlinear]

[^nonlinear]: Inverse models can be formed also for some nonlinear systems. [ModelingToolkit.jl](https://mtk.sciml.ai/dev/) is particularly well suited for inverting models due to its acausal nature.

The filter $Q(z)$ acts to make the procedure robust w.r.t. noise and modeling errors. $Q$ has a final say over what frequencies appear in $a$ and it's good to choose $Q$ with low-pass properties. $Q$ will here be applied in zero-phase mode, so the effective transfer function will be $Q(z)Q(z̄)$.

```@example HEURISTIC_ILC
z = tf("z", Ts)
Q = c2d(tf(1, [0.05, 1]), Ts)
# L = 0.9z^1 # A more conservative and heuristic choice
L = 0.5inv(Gr) # Make the scaling factor smaller to take smaller steps

alg = HeuristicILC(Q, L, :ref)
nothing # hide
```

A theorem due to Norrlöf says that for the ILC iterations to converge, one needs to satisfy
$$| 1 - LG | < |Q^{-1}|$$
which we can verify by looking at the Bode curves of the two sides of the inequality

```@example HEURISTIC_ILC
ilc_theorem(alg, Gr, tf(Gract))
```


Above, we plotted this curve also for the actual dynamics. This is of course not possible in a real scenario where this is unknown, but one could plot it for multiple plausible models and verify that they are all below the boundary. See [Uncertainty modeling using RobustAndOptimalControl.jl](https://juliacontrol.github.io/RobustAndOptimalControl.jl/dev/uncertainty/) for guidance on this. Looking at the stability condition, it becomes obvious how making $Q$ small where the model is uncertain is beneficial for robustness of the ILC scheme.

## ILC iteration

The next step is to implement the ILC scheme and run it:
    
```@example HEURISTIC_ILC
prob = ILCProblem(; r, Gr, Gu)
sol = ilc(prob, alg)
plot(sol)
```

When running on the model, the result looks very good.
We see that the tracking error in the last plot decreases rapidly and is much smaller after only a couple of iterations. We also note that the adjusted reference $r+a$ has effectively been phase-advanced slightly to compensate for the lag in the system dynamics. This is an effect of the acausal filtering due to $L = G_C^{-1}$.


How does it work on the "actual" dynamics? To simulate the effect of plant-model mismatch, one may provide a different instance of the ILCProblem using the `actual` keyword argument which is used to simulate the plant response. The ILC update will be performed using the plant model from `prob`, while simulated data will be acquired from the models in `actual`.
```@example HEURISTIC_ILC
actual = ILCProblem(; r, Gr=Gract, Gu=Guact)
sol = ilc(prob, alg; actual)
plot(sol)
```
The result is subtly worse, but considering the rather big model error the result is still quite good. 


## Assessing convergence under uncertainty
We can attempt at modeling the uncertainty we have in the plant using uncertain numbers from [MonteCarloMeasurements.jl](https://github.com/baggepinnen/MonteCarloMeasurements.jl). This will allow us to asses whether our chosen filter fulfil the convergence criteria for all possible realizations of the uncertain plant.


```@example HEURISTIC_ILC
using MonteCarloMeasurements
unsafe_comparisons(true)
Pact = double_mass_model(
    Jl = 0.5..1.5, # ± 50% load uncertainty
    c1 = 0.8..1.2, # ± 20% transmission damping uncertainty
    c2 = 0.8..1.2, # ± 20% load damping uncertainty
) 
Gract = c2d(feedback(Pact*C), Ts)
w = exp10.(-2:0.01:2)
ilc_theorem(alg, Gr, tf(Gract); w)
```
In this case, it looks like we're good to go!

Learn more about modeling uncertainty in control systems under [RobustAndOptimalControl: Uncertainty modeling](https://juliacontrol.github.io/RobustAndOptimalControl.jl/dev/uncertainty/).

## Docstring
    
```@docs
HeuristicILC
```