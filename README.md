# IterativeLearningControl2

[![Build Status](https://github.com/baggepinnen/IterativeLearningControl2.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/baggepinnen/IterativeLearningControl2.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Documentation, stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://baggepinnen.github.io/IterativeLearningControl2.jl/stable)
[![Documentation, latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://baggepinnen.github.io/IterativeLearningControl2.jl/dev)

[Iterative-Learning Control (ILC)](https://slunik.slu.se/kursfiler/TE0010/10095.1213/REG2_ILCReview.pdf) for linear and nonlinear systems.

## What is ILC?

ILC can be thought of as either
- a simple reinforcement-learning (RL) strategy, or
- a method to solve open-loop optimal control problems.

ILC is suitable in situations where a *repetitive task* is to be performed multiple times, and disturbances acting on the system are also repetitive and predictable but  may be unknown. Multiple versions of ILC exists, of which we support a few that are listed below. When ILC iterations are performed by running experiments on a physical system, ILC resembles episode-based reinforcement learning (or adaptive control), while if a model is used to simulate the experiments, we can instead think of ILC as a way to solve optimal control problems (trajectory optimization).

See [the documentation](https://baggepinnen.github.io/IterativeLearningControl2.jl/dev) for more details.