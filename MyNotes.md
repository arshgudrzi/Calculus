# Calculus

The Foundation for every neural network. Here's why we should actually care:
- Training neural networks = finding minimum of loss function
- Gradient descent = calculus in action
- Signal processing = understanding rate of change
- Optimization problems = all calculus

My entire AI automation runs on this stuff or something like that haha.

---

## LEVEL 1: Derivatives - The Rate of Change

**What is it?** Well, that just means the rate of change. For example, how fast the car is going. And if you look closely, you see that we had to shrink time-to-distance down to zero so that we know how fast we are going at that EXACT moment. Not the average. Not the trip. THIS moment.

Imagine driving a car:
- Distance: how far you've gone
- Speed (derivative): how fast you are going RIGHT NOW

It is NOT like distance / time, because that is the average speed. The derivative is the instantaneous speed. The "right now" speed. The "I am speeding and the camera just caught me" speed, duh.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simple function: f(x) = x^2
def f(x):
    return x**2

# Derivative: f'(x) = 2x
# Power Rule: d/dx[x^n] = n * x^(n-1)
# Here n=2, so f'(x) = 2*x^(2-1) = 2x --> duh, just bring the exponent down and subtract 1
def f_prime(x):
    return 2*x

# Visualize
x = np.linspace(-3, 3, 100)
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(x, f(x), label='f(x) = x²')
plt.grid(True)
plt.title('Original Function')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, f_prime(x), label="f'(x) = 2x", color='red')
plt.grid(True)
plt.title('Derivative (Rate of Change)')
plt.legend()

plt.tight_layout()
plt.show()
```

### Why I Care (don' know about the people who read this)
This is literally how gradient descent works:
```python
# Gradient descent in one line
w = w - learning_rate * gradient
#           ↑                ↑
#     how big of step    which direction (derivative!)
```

> I obviously learned this with AI and this is the example they always give. I am kind of a copy cat haha. But the concept is real.

---

## LEVEL 2: The Power Rule (90% of What We Need)

### The Pattern
```
f(x) = x^n
f'(x) = n * x^(n-1)
```

That's it. Bring the exponent down, multiply, reduce exponent by 1. You now know 90% of what people use calculus for. Congratulations, you are practically a mathematician, or something like that.

### DO THIS NOW - Gradient Descent on a Simple Function
```python
import numpy as np
import matplotlib.pyplot as plt

# Function: f(x) = (x-3)^2 + 1  (parabola, minimum at x=3, value=1)
def f(x):
    return (x - 3)**2 + 1

# Derivative: f'(x) = 2(x-3)
# This tells us: which direction is uphill, and how steep
def f_prime(x):
    return 2 * (x - 3)

# Gradient descent algorithm
x = 0.0          # Start at x=0 (far from minimum at x=3)
learning_rate = 0.1  # How big of a step to take each iteration
history = [x]    # Track our path (for visualization)

for i in range(50):  # 50 iterations
    gradient = f_prime(x)             # Calculate slope at current position
    x = x - learning_rate * gradient  # Move opposite to gradient (downhill)
    history.append(x)

    if i < 5 or i % 10 == 0:  # % is modulo operator (remainder after division)
        print(f"Step {i}: x={x:.4f}, f(x)={f(x):.4f}, gradient={gradient:.4f}")
        # :.4f formats float to 4 decimal places

# Visualize the journey
x_plot = np.linspace(-1, 7, 100)
plt.figure(figsize=(10, 6))

plt.plot(x_plot, f(x_plot), 'b-', label='f(x)')
# 'b-' means: blue solid line

plt.plot(history, [f(x) for x in history], 'ro-', label='GD path', markersize=4)
# 'ro-' means: red circles connected by lines

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient Descent Finding Minimum')
plt.legend()
plt.grid(True)
plt.show()

print(f"\nFinal x: {x:.4f} (should be close to 3)")
```

---

## LEVEL 3: Chain Rule - The Most Important Rule

### The Concept (The "Physical Chain")
Imagine a physical system where one thing pulls another.
1. You pull a lever using your hand (Input `x`).
2. The lever pulls a rope (Intermediate `u`).
3. The rope lifts a heavy weight (Output `y`).

If pulling the lever **1 inch** moves the rope **2 inches** (`du/dx = 2` — lever ratio),
And moving the rope **1 inch** lifts the weight **3 inches** (`dy/du = 3` — pulley ratio),

**How much does the weight move when you pull the lever?**
It moves `2 × 3 = 6` inches. You just multiply the rates because the effects **amplify** through each link in the chain. Duh.

**That is the Chain Rule.**

$$ dy/dx = dy/du \times du/dx $$

### Step-by-Step Example: f(x) = (3x + 1)²
Don't just memorize the rule. Break it down into links. This function is two steps:
1. **Inside**: u = 3x + 1
2. **Outside**: y = u² (where u is the inside stuff)

**Step 1: Find the rate of the Inside**
How fast does u change if x changes?
Derivative of 3x + 1 is **3**.
`du/dx = 3`

**Step 2: Find the rate of the Outside**
How fast does y change if u changes?
Derivative of u² is **2u**.
`dy/du = 2u`

**Step 3: Multiply them (The Chain)**
Total rate = Outside Rate × Inside Rate
f'(x) = (2u) × (3) = 6u

**Step 4: Swap u back**
Put "(3x+1)" back where "u" was.
f'(x) = 6(3x + 1)

Boom. Done. Chain rule executed. Arash, you brilliant person you (ok.... I just got hyped doing something that some highschooler does, shame on me. lame me. (ashamed noises)).

### The Intuition
Why multiply?
- The inner function (3x+1) speeds things up by **3x**
- The outer function (u²) speeds things up by **2u** times
- The total speed-up is the product of the individual speed-ups

Think of it like the lever and rope: each link amplifies or dampens what the previous link did. You just multiply them. That's literally it.

### Use Case in AI
Every neural network is just nested functions (layers):
```python
# Network: input → layer1 → layer2 → output
output = f(g(h(input)))

# Backpropagation is just chain rule:
d_output/d_input = f'(g(h)) * g'(h) * h'(input)
#                  ↑        ↑        ↑
#              Last Layer * Middle * First
```

> Wait a minute. You just understood backpropagation. The thing that makes ALL neural networks learn. I did not have the energy to write the 3 lines above, AI did it, but you can think of the credit for me. danke.

### DO THIS NOW - Backpropagation from Scratch
```python
import numpy as np

# Simple network: input → hidden → output
# Architecture: 1 input → 2 hidden neurons → 1 output

# ReLU activation: max(0, x) - introduces non-linearity
def relu(x):
    return np.maximum(0, x)

# Derivative of ReLU: 1 if x>0, else 0 --> please read my algebra notes. because most of my notes are connected together!
# Here's the intuition for why this matters in backpropagation:
# Think of ReLU as a gate. If a neuron was "on" (x > 0), the gradient flows
# through it freely (derivative = 1). If it was "off" (x <= 0), the gradient
# gets BLOCKED (derivative = 0). This is how the network figures out which
# neurons actually contributed to the error. Only the active ones get updated.
# Dead neurons get ignored. Makes total sense when you think about it, duh.
def relu_derivative(x):
    return (x > 0).astype(float)  # Comparison returns bool, convert to 0.0/1.0

# Initialize tiny network weights
W1 = np.array([[0.5], [0.3]])   # Hidden layer weights
b1 = np.array([0.1, 0.2])       # Hidden layer biases

W2 = np.array([[0.4, 0.6]])     # Output layer weights
b2 = np.array([0.1])            # Output layer bias

# ========== FORWARD PASS ==========
x = np.array([2.0])

# Hidden layer
z1 = W1 @ x + b1    # Linear: weights × input + bias
                     # Result: [0.5*2 + 0.1, 0.3*2 + 0.2] = [1.1, 0.8]
h = relu(z1)        # Activation: h = [1.1, 0.8] (both positive, unchanged)

# Output layer
z2 = W2 @ h + b2    # [0.4, 0.6] @ [1.1, 0.8] + 0.1 = 1.02
y = z2[0]           # Extract scalar from array

print(f"Forward pass: x={x[0]:.2f} → y={y:.2f}")

# ========== COMPUTE LOSS ==========
target = 3.0
loss = (y - target)**2  # Mean squared error: (prediction - target)^2
# Loss = (1.02 - 3)^2 = 3.92 --> the network is way off, embarrassing honestly

# ========== BACKWARD PASS (CHAIN RULE!) ==========
# Goal: compute ∂Loss/∂W1 and ∂Loss/∂W2
# Work backwards through the network

# Gradient of loss with respect to output
# d/dy[(y - target)^2] = 2(y - target)
dloss_dy = 2 * (y - target)  # = 2 * (1.02 - 3) = -3.96

dy_dz2 = 1  # y = z2, so dy/dz2 = 1

dz2_dW2 = h    # z2 = W2 @ h, so ∂z2/∂W2 = h (the input to this layer)
dz2_dh = W2    # z2 = W2 @ h, so ∂z2/∂h = W2

# CHAIN RULE: ∂Loss/∂W2 = (∂Loss/∂y) × (∂y/∂z2) × (∂z2/∂W2)
dloss_dW2 = dloss_dy * dy_dz2 * dz2_dW2

# Gradient through ReLU activation
dh_dz1 = relu_derivative(z1)   # h = relu(z1)

dz1_dW1 = x    # z1 = W1 @ x, so ∂z1/∂W1 = x

# CHAIN RULE: ∂Loss/∂W1 = (∂Loss/∂y) × (∂y/∂z2) × (∂z2/∂h) × (∂h/∂z1) × (∂z1/∂W1)
dloss_dW1 = dloss_dy * dy_dz2 * dz2_dh.T * dh_dz1 * dz1_dW1

print(f"\nGradients (how to change weights):")
print(f"dLoss/dW2 = {dloss_dW2}")
print(f"dLoss/dW1 = {dloss_dW1.flatten()}")

# ========== UPDATE WEIGHTS (GRADIENT DESCENT) ==========
lr = 0.01
W2_new = W2 - lr * dloss_dW2.reshape(W2.shape)
W1_new = W1 - lr * dloss_dW1

print(f"\nOriginal loss: {loss:.4f}")

# Forward pass with new weights to verify
z1_new = W1_new @ x + b1
h_new = relu(z1_new)
z2_new = W2_new @ h_new + b2
y_new = z2_new[0]
loss_new = (y_new - target)**2

print(f"New loss: {loss_new:.4f}")
print(f"Loss decreased: {loss > loss_new}")
```

> Wait a minute, did you just see? You little brilliant scientist. You just implemented backpropagation from scratch. The thing PyTorch and TensorFlow do automatically in one line — you did it by hand. I am proud of you the person who is reading this note. Unless you just copy-pasted without understanding, in which case... well, that is also on me for not putting more barriers haha.

---

## LEVEL 4: Partial Derivatives - Multiple Variables

### What It Is

When a function has multiple inputs: f(x, y, z)

Partial derivative = derivative with respect to ONE variable, treating all others as constants.

Symbol: ∂f/∂x (instead of df/dx)

Think of it like this: you're standing on a hilly landscape. The partial derivative with respect to x tells you how steep the hill is if you walk ONLY in the x direction. The partial w.r.t. y tells you how steep if you walk ONLY in y. The gradient combines both into "the steepest direction overall". Makes sense? It should, duh.

```python
import numpy as np

# f(x, y) = x^2 + 3xy + y^2

# ∂f/∂x = 2x + 3y  (treat y as constant, same old power rule on x terms)
# ∂f/∂y = 3x + 2y  (treat x as constant)

def f(x, y):
    return x**2 + 3*x*y + y**2

def df_dx(x, y):
    return 2*x + 3*y

def df_dy(x, y):
    return 3*x + 2*y

# Gradient = vector of all partial derivatives
# This is the direction of steepest ascent. Flip it (negative) = steepest descent
def gradient(x, y):
    return np.array([df_dx(x, y), df_dy(x, y)])

x, y = 1.0, 2.0
print(f"f({x}, {y}) = {f(x, y)}")
print(f"Gradient at ({x}, {y}) = {gradient(x, y)}")
```

### Why You Care
Neural networks have millions of parameters. You need to find ∂Loss/∂w for EVERY single weight w. Every single one. That is what autograd in PyTorch does automatically under the hood. Partial derivatives. Everywhere. All the time. At massive scale.

### DO THIS NOW - 2D Gradient Descent
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function: f(x,y) = x^2 + y^2 (bowl shape, minimum at origin)
def f(x, y):
    return x**2 + y**2

def gradient(x, y):
    return np.array([2*x, 2*y])

# Gradient descent in 2D
position = np.array([5.0, 5.0])  # Start far from minimum
lr = 0.1
path = [position.copy()]

for _ in range(50):
    grad = gradient(position[0], position[1])
    position = position - lr * grad
    path.append(position.copy())

path = np.array(path)

# Visualize
fig = plt.figure(figsize=(14, 5))

# 3D surface
ax1 = fig.add_subplot(121, projection='3d')
x_grid = np.linspace(-6, 6, 50)
y_grid = np.linspace(-6, 6, 50)
X, Y = np.meshgrid(x_grid, y_grid)
Z = f(X, Y)
ax1.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis')
ax1.plot(path[:, 0], path[:, 1], [f(p[0], p[1]) for p in path],
         'ro-', linewidth=2, markersize=4)
ax1.set_title('3D View')

# Contour plot
ax2 = fig.add_subplot(122)
ax2.contour(X, Y, Z, levels=20)
ax2.plot(path[:, 0], path[:, 1], 'ro-', linewidth=2, markersize=4)
ax2.set_title('Top-Down View')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.grid(True)

plt.tight_layout()
plt.show()

print(f"Started at: {path[0]}")
print(f"Ended at: {path[-1]} (should be close to [0, 0])")
```

**Realization**: This is how you train networks with millions of parameters. Find gradient, take step, repeat. That is it. Billions of dollars of AI infrastructure doing exactly this loop. Kind of humbling honestly. Or embarrassing for the industry. I don't know.

---

## LEVEL 5: Integrals - Adding Up Infinitely Small Pieces

### What It Is
Integral = sum of infinitely many infinitely small pieces.

If derivative = rate of change, then integral = total accumulation.

```
∫ f(x) dx = "area under curve f(x)"
```

### The Connection
Derivatives and integrals are opposites (Fundamental Theorem of Calculus):
```
If F'(x) = f(x), then:
∫[a to b] f(x)dx = F(b) - F(a)
```

They undo each other. Like multiplication and division, but for functions. The people who discovered this probably felt like gods and honestly, fair enough.

### The Why (Accumulation)
If derivatives slice a shape into thin strips (rates), integrals glue them back together.
- **Physics**: Integral of Speed = Distance. (Summing up "how fast you went" × "for how long" at every instant gives total distance)
- **Geometry**: Area under the curve. We use rectangles `height × width` (`f(x) × dx`) and make them infinitely thin to fill the shape perfectly

### Why You Care
- Probability distributions (area under curve = probability)
- Expected values = integral of x × probability
- Physics simulations = integrate forces to get motion
- Signal processing = integrate to smooth/filter

### DO THIS NOW - Monte Carlo Integration
```python
import numpy as np
import matplotlib.pyplot as plt

# Integrate f(x) = x^2 from 0 to 1
# Analytical answer: ∫ x^2 dx = x^3/3 |_0^1 = 1/3 ≈ 0.333
# We could compute this analytically but that's boring, let's throw darts instead

def f(x):
    return x**2

# Monte Carlo method: random sampling
# Throw darts at the unit square, count how many land under the curve
n_samples = 10000
x_random = np.random.uniform(0, 1, n_samples)
y_random = np.random.uniform(0, 1, n_samples)

# Count points under curve
under_curve = y_random < f(x_random)
area_estimate = under_curve.sum() / n_samples

print(f"Analytical integral: {1/3:.6f}")
print(f"Monte Carlo estimate: {area_estimate:.6f}")
print(f"Error: {abs(area_estimate - 1/3):.6f}")

# Visualize
plt.figure(figsize=(10, 6))
x = np.linspace(0, 1, 100)
plt.plot(x, f(x), 'b-', linewidth=2, label='f(x) = x²')
plt.fill_between(x, f(x), alpha=0.3)
plt.scatter(x_random[under_curve], y_random[under_curve],
           c='green', s=1, alpha=0.5, label='Under curve')
plt.scatter(x_random[~under_curve], y_random[~under_curve],
           c='red', s=1, alpha=0.5, label='Above curve')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.title('Monte Carlo Integration')
plt.show()
```

**Cool Insight**: Monte Carlo is how you estimate integrals you can't solve analytically. Used everywhere in ML (MCMC sampling, RL, etc.). Basically: statistics + randomness + counting. Duh. You can try this same idea with a circle to estimate π, which sounds dumb but actually works, and actually makes me feel that math is magic or something like that.

---

> And remember, as in hacking, you don't learn to hack something specific. You just learn the tools and how they work. Your intuition and creativity is the weapon.
>
> So as in fighting sports you learn the moves. But you don't learn to beat some athlete. You just use the movements and refine them to beat the opponent. But I guess we all are just not into fighting sports if we are studying calculus haha. Tschüssi or something cuter.

---

## Tests: Let's See If You Were Actually Paying Attention

### Easy:
1. Use the power rule to differentiate `f(x) = 3x^4 + 2x^2 - 7x + 5`. Just write out the answer, duh.
2. Run gradient descent from scratch to find the minimum of `f(x) = (x - 7)^2 + 2`. The minimum is obviously at x=7 with value=2, but the point is to watch the algorithm FIND it.

### Medium:
3. Implement 2D gradient descent that finds the minimum of `f(x, y) = (x-2)^2 + (y-3)^2`. Visualize the path.
4. Estimate π using Monte Carlo. Hint: throw random points at a 2×2 square centered at origin, count how many land inside the unit circle.

### Hard:
5. Implement backpropagation from scratch for a 2-layer network. Train it on XOR (the classic problem that broke single-layer perceptrons and caused an "AI winter", which is dramatic but kind of cool).
6. Optional and probably overkill honestly: Build the whole gradient descent optimizer with momentum and see if it converges faster. I don't know if I will ever do this one either haha.

---

## Solutions (because I am nice like that, but also because I know you will look at them immediately, you loser haha):

### Solution 1 — Power Rule

```python
# f(x) = 3x^4 + 2x^2 - 7x + 5
# Apply power rule term by term: bring down exponent, reduce by 1

# f'(x) = 3*4*x^3 + 2*2*x^1 - 7*1*x^0 + 0
# f'(x) = 12x^3 + 4x - 7

# Verify with numpy (numerical differentiation)
import numpy as np

def f(x):
    return 3*x**4 + 2*x**2 - 7*x + 5

def f_prime_analytical(x):
    return 12*x**3 + 4*x - 7  # our hand-calculated derivative

def f_prime_numerical(x, h=1e-7):
    return (f(x + h) - f(x)) / h  # definition of derivative, limit as h→0

x_test = 2.0
analytical = f_prime_analytical(x_test)
numerical = f_prime_numerical(x_test)

print(f"Analytical f'({x_test}) = {analytical}")    # 96 + 8 - 7 = 97
print(f"Numerical  f'({x_test}) = {numerical:.4f}") # should be ≈ 97
print(f"Match: {abs(analytical - numerical) < 0.001}")
```

> Did you get it? did you enjoy the latex show off, Arash? no? ok. but the important thing is the numerical verification trick — it's useful whenever you want to check that your gradient math is right. PyTorch uses a similar thing called gradcheck. Now you are basically PyTorch core. I am joking, but you get the idea.

---

### Solution 2 — Gradient Descent Finding Minimum of (x-7)²+2

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return (x - 7)**2 + 2

def f_prime(x):
    return 2 * (x - 7)

x = 0.0           # Start far from x=7
learning_rate = 0.1
history = [x]

for i in range(60):
    gradient = f_prime(x)
    x = x - learning_rate * gradient
    history.append(x)

    if i < 5 or i % 10 == 0:
        print(f"Step {i}: x={x:.4f}, f(x)={f(x):.4f}")

x_plot = np.linspace(-1, 12, 200)
plt.figure(figsize=(10, 6))
plt.plot(x_plot, f(x_plot), 'b-', label='f(x) = (x-7)² + 2')
plt.plot(history, [f(xi) for xi in history], 'ro-', label='GD path', markersize=4)
plt.axvline(x=7, color='green', linestyle='--', label='True minimum x=7')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient Descent Rolling Down to x=7')
plt.legend()
plt.grid(True)
plt.show()

print(f"\nFinal x: {x:.4f} (should be ≈ 7.0)")
print(f"Final f(x): {f(x):.4f} (should be ≈ 2.0)")
```

> You started at 0 and the algorithm rolled all the way to 7. That is literally how neural networks learn. Not magic. Just derivative. Step. Repeat. And we give it a cool name like "training". haha.

---

### Solution 3 — 2D Gradient Descent on f(x,y) = (x-2)²+(y-3)²

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x, y):
    return (x - 2)**2 + (y - 3)**2

def gradient(x, y):
    # ∂f/∂x = 2(x-2)
    # ∂f/∂y = 2(y-3)
    return np.array([2*(x - 2), 2*(y - 3)])

position = np.array([8.0, -2.0])  # Start far from minimum (2, 3)
lr = 0.1
path = [position.copy()]

for _ in range(60):
    grad = gradient(position[0], position[1])
    position = position - lr * grad
    path.append(position.copy())

path = np.array(path)

fig = plt.figure(figsize=(14, 5))

ax1 = fig.add_subplot(121, projection='3d')
x_grid = np.linspace(-1, 10, 60)
y_grid = np.linspace(-4, 8, 60)
X, Y = np.meshgrid(x_grid, y_grid)
Z = f(X, Y)
ax1.plot_surface(X, Y, Z, alpha=0.5, cmap='viridis')
ax1.plot(path[:, 0], path[:, 1], [f(p[0], p[1]) for p in path],
         'ro-', linewidth=2, markersize=4)
ax1.set_title('3D View (rolling to minimum)')

ax2 = fig.add_subplot(122)
contours = ax2.contour(X, Y, Z, levels=25, cmap='viridis')
ax2.plot(path[:, 0], path[:, 1], 'ro-', linewidth=2, markersize=5)
ax2.plot(2, 3, 'g*', markersize=15, label='True minimum (2, 3)')
ax2.set_title('Top-Down View (Contour Map)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

print(f"Started at: {path[0]}")
print(f"Ended at: {path[-1]} (should be ≈ [2.0, 3.0])")
```

> If you stare at the contour plot long enough you will feel like you understand why ML people always draw loss landscapes. That is exactly what this is. The algorithm rolls down the bowl to the minimum. When you have 175 billion parameters (like GPT-4 or whatever), it is the same idea but in 175-billion-dimensional space. Nobody can visualize that. Not even OpenAI. So don't feel bad, duh.

---

### Solution 4 — Monte Carlo Estimation of π

```python
import numpy as np
import matplotlib.pyplot as plt

# Idea: Unit circle has area π*r² = π*1² = π
# Square from -1 to 1 in both axes has area 4
# If we throw random darts at the square:
# probability of hitting the circle = π/4
# so: π ≈ 4 × (darts_in_circle / total_darts)

n_samples = 100000
x_rand = np.random.uniform(-1, 1, n_samples)
y_rand = np.random.uniform(-1, 1, n_samples)

# Point is inside unit circle if x^2 + y^2 <= 1
in_circle = (x_rand**2 + y_rand**2) <= 1

pi_estimate = 4 * in_circle.sum() / n_samples

print(f"Real π:       {np.pi:.6f}")
print(f"Our estimate: {pi_estimate:.6f}")
print(f"Error:        {abs(pi_estimate - np.pi):.6f}")

# Visualize
plt.figure(figsize=(8, 8))
plt.scatter(x_rand[in_circle], y_rand[in_circle],
            c='green', s=0.5, alpha=0.5, label='Inside circle')
plt.scatter(x_rand[~in_circle], y_rand[~in_circle],
            c='red', s=0.5, alpha=0.5, label='Outside circle')

theta = np.linspace(0, 2*np.pi, 300)
plt.plot(np.cos(theta), np.sin(theta), 'b-', linewidth=2, label='Unit circle')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.gca().set_aspect('equal')
plt.title(f'Monte Carlo π ≈ {pi_estimate:.4f} (n={n_samples})')
plt.legend()
plt.grid(True)
plt.show()
```

> Well you know, I kind of feel we just approximated one of the most famous constants in mathematics by throwing random darts. And it works. And that is a little insane honestly. The more darts, the better the estimate. This is Monte Carlo method, used everywhere in finance, physics, ML. Basically when you can't compute something exactly, you just... estimate it randomly. Beautiful chaos. Tschüssi.

---

### Solution 5 — Backpropagation on XOR from Scratch

```python
import numpy as np

# XOR problem: the classic one that destroyed single-layer networks
# Input  → Output
# [0, 0] → 0
# [0, 1] → 1
# [1, 0] → 1
# [1, 1] → 0
# A single line can't separate this. You need a hidden layer. That is the whole point.

# Sigmoid: smooth version of step function, between 0 and 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)  # derivative of sigmoid is sigmoid*(1-sigmoid), convenient haha

# XOR training data
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])  # shape (4, 2)

y = np.array([[0], [1], [1], [0]])  # shape (4, 1)

# Network: 2 inputs → 4 hidden neurons → 1 output
np.random.seed(42)  # reproducibility, because I am kind of a copy cat like that haha

W1 = np.random.randn(2, 4) * 0.5   # (2 inputs → 4 hidden)
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.5   # (4 hidden → 1 output)
b2 = np.zeros((1, 1))

lr = 0.5
losses = []

for epoch in range(10000):
    # ========== FORWARD PASS ==========
    z1 = X @ W1 + b1          # (4, 2) @ (2, 4) = (4, 4)
    h = sigmoid(z1)            # (4, 4)

    z2 = h @ W2 + b2           # (4, 4) @ (4, 1) = (4, 1)
    y_hat = sigmoid(z2)        # (4, 1)  --> final predictions

    # ========== LOSS ==========
    loss = np.mean((y_hat - y)**2)  # Mean squared error
    losses.append(loss)

    # ========== BACKWARD PASS (CHAIN RULE!) ==========
    # Gradient of loss w.r.t. y_hat
    dloss_dyhat = 2 * (y_hat - y) / len(X)

    # Output layer gradients
    dyhat_dz2 = sigmoid_derivative(z2)
    delta2 = dloss_dyhat * dyhat_dz2   # (4, 1)

    dW2 = h.T @ delta2                 # (4, 4).T @ (4, 1) = (4, 1)
    db2 = delta2.sum(axis=0, keepdims=True)

    # Hidden layer gradients
    dh_dz1 = sigmoid_derivative(z1)
    delta1 = (delta2 @ W2.T) * dh_dz1  # Chain rule through W2

    dW1 = X.T @ delta1                 # (2, 4).T @ (4, 4) = (2, 4)
    db1 = delta1.sum(axis=0, keepdims=True)

    # ========== UPDATE WEIGHTS ==========
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if epoch % 2000 == 0:
        print(f"Epoch {epoch}: loss = {loss:.6f}")

# Final predictions
z1 = X @ W1 + b1
h = sigmoid(z1)
z2 = h @ W2 + b2
y_hat = sigmoid(z2)

print("\nFinal predictions:")
for i, (xi, yi, pred) in enumerate(zip(X, y, y_hat)):
    print(f"  Input {xi} → Target {yi[0]} → Predicted {pred[0]:.4f} → Rounded {round(pred[0])}")

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.title('XOR Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.yscale('log')
plt.show()
```

> Mind Blown: You just trained a neural network from scratch on XOR. No PyTorch. No TensorFlow. Just numpy and chain rule. The thing that caused an entire "AI winter" in the 1960s when people thought it was impossible — you just solved it in 50 lines. If you understand every line above, you understand deep learning at a fundamental level. Go touch grass or something. You earned it. I used arch btw, this was completely irrelevant, but I am seeking social validation through arch. duh.

---
ummmm... at the end, thanks for being with me yo. enjoy the notes
