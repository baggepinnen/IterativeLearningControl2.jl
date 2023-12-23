# Manual ILC iterations
To perform ILC on a physical system, we need to repeatedly apply an ILC signal to our system under test, record the result, and compute a new ILC signal. The function [`ilc`](@ref) that is used in the examples in this documentation performs all these steps internally, but using a simulation model of the system. To perform these steps manually, we need to do the following:
1. Initialize the algorithm using the function [`init`](@ref). Also choose an initial ILC signal, typically all zeros, but may also be chosen as the result of a few ILC iterations on a simulated system.
2. Perform an experiment on the system under test and record the resulting output ``y``
3. Compute the tracking error ``e = r - y``
4. Compute a new ILC input signal ``a`` using the function [`compute_input`](@ref)
5. Repeat steps 2-4 until the desired performance is achieved.
6. Enjoy sweet tracking performance, maybe also tell your mother about it.

The function `workspace = init(prob, alg)` takes the specification of the ILC problem and the chosen algorithm and returns a `workspace` object with problem-specific things the algorithm needs to run. 

The function `a = compute_input(prob, alg, workspace, a, e)` takes the problem, algorithm, the workspace object created by `init`, the previous ILC input signal ``a_k`` and the tracking error ``e = r - y`` and returns a new ILC input signal ``a_{k+1}``.

## Docstrings
```@docs
init
compute_input
```