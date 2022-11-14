# SPH simulation code
## Introduction
- Smoothed Particle Hydrodynamics code for 2D examples : Dam break problem
- Weakly compressible assumption
## Detail
- Density Approximation : use continuity equation 
- Pressure Approximation : use equation of state 
- Boundary condition : elastic collision (100% reflection) applied

## Result
<div>
    <p>Simulation result for 2D(left : single, right : parallel)</p>
    <p float = "left">
        <img src="/result/simulation.gif"  width="320" height="196">
        <img src="/result/simulation-parallel.gif"  width="320" height="196">
    </p>
</div>

## Reference
- Based on SNU Lectures(고성능 입자법 기반 시뮬레이션)
- github code : https://github.com/AlexandreSajus/Python-2D-SPH