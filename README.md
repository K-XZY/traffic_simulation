
# Traffic Simulation
This is a traffic model that models the microscopic behavior of individual drivers, to yield macroscopic phenomenons (e.g. Congestion). We will include stochasticity, statistics and delayed ordinary differential equations with high dimension yet sparse causal connections. 

# Theoretical Backgrounds
## System overview
We have a system $(T,P,\Phi)$ Where:
- $T = \mathbb R^+$ is the monoid of time, or the compute step. 
- $P\subset \mathbb R^{2*n}$  is the set of states, where each state $p\in P$, represents an instance of $[x_1,..., x_n, \dot x_1,..., \dot x_n] = [x,\dot x]$.
- $\Phi$ is the state transition function, that maps $P$ to $P$.

Semantically, each feature of $p$ represents a feature of a driver. More specifically, $[x_i,\dot x_i]$ are features of the $i_{th}$ driver. 

The system dynamics $\Phi$ follows the following delayed ODE and integral:
$$
\dot x(t+\tau) = f(x(t) | \theta)
$$
$$
x(t) = \int_0 ^ {t} \dot x(t)  dt
$$
The choice of $f$ and $\theta$ essentially govern's how the driver react's to other cars, we will demonstrate it's derivation later. Remember that, $x$ stands for the vector $[x_1,...,x_n]$, as part of the model state.

We will sample part of the parametrization $\theta$ as experiment set up, and part of it during the experimentation as noise and driver behavior.

For our simulation, we will extract statistics about individual driving speed, as well as the flux (throughput) of the road, along with with the center of mass of all cars over time. We will also extract discrete information, such as the number of times a car halts as they reach near 0 speed, and how certain parameters in $\theta$ is correlated with these statistics, studied via a monte-carlo experiment.


## Assumptions 
1. Identical control algorithm for all drivers, with only specific parameters differs.
2. Drivers only react to difference and distance.
3. Cars travels on only 1 lane. i.e. $x_i-x_j = x_i - x_k \implies j = k$.
4. Cars do not pass through the car before it. i.e. $x_i\lt x_{i+1} - L_{i+1}$ where $L_{i+1}$ is the length of the car $i$. 
5. Driver's only react to the 1 car before it. (i.e. Sparse causal connections among state variables). i.e. $\dot x_i (t+\tau) = f(x_i(t),x_{i+1}(t))$.
6. Deceleration inputs converts directly to velocity. (e.g. No slippery floor, or lose of energy when controlling the car). i.e. we can directly model velocity.
7. Rational drivers, that decelerates when they are too close to others before them, and decelerates if they are driving faster than cars before them.
8. The parametrization of $f$ with $\theta$ is a time-independent. However, stochasticity can be, and will be included for some of our simulations.

## Model in detail
### The ODE of Car Agent:
We can start by stating that:
$$
\dot x_i(t+\tau) = f(D_i(t) | \theta_i)
$$
where $D^d_i(t) = x_{i+1} - x_{i}$. i.e. We want the driving speed to be related to difference in distance to cars before you. You can view $D$ as both difference and contribution to deceleration, as stated in the assumption. We also want all cars to behave we the same algorithm $f$ parametrized by $\theta_i$.

Now, it is time to find out what exactly is this $f$ and $\theta$. We want the cars to decelerate. Since usually cars travel at a constant speed, we set a distance $D_{c}$ that is the critical distance which the car will start to react and decelerate. Therefore, when $D_i \ge D_c$, we have $\dot x_i(t+\tau) = V_{max}$. Similarly, when we are very close to the car before us with $D_i \le L$, we want to completely slow down with $\dot x_i (t+\tau) = 0$. Making $f$ a 3-part function. 

We construct the following equation to specify f:
$$
\dot x(t) = C(V_{max},D_{c},L)\ln(\frac{D(t)}{L})
$$
More specifically, we choose constant $C = V_{max} \ln (\frac{D_c}{L})^{-1}$ as this ensures the $\ln$ terms become $1$ at $D(t) = D_c$, giving us $\dot x = V_{max}$ when we are right at the critical distance. This choice of $C$ also guarantee that when $D(t) = L$ we have $\dot x = 0$. However, in computation we must add a small $\epsilon$ to the term, like $(\ln (D_c/L + \epsilon)^-1$.

We can see this on the plot:  (A plot of the curve of x dot and D)

To make the simulation even more realistic, we should also include a bias term to the driver's estimate on distance. That is reducing $D(t)$ by a small $b$, indicating that some driver's might falsely estimate how far a car is before them. In the future, we can consider making $b$ a function of $\dot x$ as estimation failures usually escalate as we are driving faster. However, this will significantly complicate the model with only marginal effects on the macroscopic behaviors, thus not included.

### Stochasticity and Variety
 We can now conclude that the parametrization $\theta = [V_{max}, D_c, L, \tau, b]$. In reality, these terms can all be noisy, or noise (e.g. $b$). However, we will only focus on randomizing $\tau$ during the run, and treat the rest as initial conditions which can be randomly sampled when setting up the experiment trials to enable more freedom in variating agents.

The delay constant $\tau$ will be randomized by Perlin noise, which gives it higher probability to sample a value closer to its current values. Keeping the changes in delay smooth and grdual.

### Statistics
Not written yet. 
Density region can be measured as computing the center of mass of all cars. Yield a coordinate.
Flux is number of cars passing a place per period of time. Yielding a single number. The lower the more congested a place is.
Average driving speed is easy. 
Correlation is computed via a monte-carlo experiment by trying out different bias and delay values, and run many simulations to average one of the above metrics, and plot the relaitonship.

### Simulation tools 
We use `Python` as the programming language. With the following libraries:
1. `Numpy` and `PyTorch` for array computation with parallelism. 
2. `Noise` for the Perlin noise generation. 
3. `Matplotlib` and `Seaborn` for visualization.
We do not use any libraries to solve the ODE, and instead implement our own solver. 
This selection of tools minimizes the complexity of the software, keeping things lightweight, open-sourced and easy to share. 


### Solving the ODEs 
The diagram below visualizes how the position $x$ in the system changes over time. Where an identical operation is applied on each node, to generate their next self. We can visualize this function as a computation graph, with a weight sharing convolution (sparse connection), a non-linearity, scale by a small time-step $\delta t$, and then adding the outcome back on itself.


This enables GPU parallelization due to the fact that variables are only linearly coupled, and can be handled with libraries like `pytorch` which itself supports `CUDA` acceleration when $n$ is really large. The draw back here, is that we would have fixed time-step $\delta t$, which might waste some compute on the time scale, but less significant when $n$ is large.

This approach allow gives a time complexity of $O(T)$ compare to the $O(N\times T)$-like complexity in most ODE solvers. 





