# Calculus — From Derivatives to Backpropagation

I always thought calculus was the boring part. The "solve the integral" homework you forget the next day. well honestly at my time, COVID was brand new and  I did not go to class and slept through the online ones!

Then I realized gradient descent — the thing that trains literally every neural network — is just calculus. And backpropagation, the thing that makes deep learning work, is just the chain rule applied repeatedly. That's it. That's the whole trick.

So I went back and actually learned it properly. With code. From first principles. No skipping the uncomfortable parts.

---

## What's In Here

One file: `MyNotes.md`

It covers calculus in 5 levels, each building on the last:

**Level 1 — Derivatives**: What a derivative actually is. The instantaneous rate of change. The "I am speeding and the camera just caught me" version of speed, not the trip average.

**Level 2 — The Power Rule**: The rule that covers 90% of what you'll ever need. Bring the exponent down, subtract 1. That's it.

**Level 3 — Chain Rule**: The most important rule. How to differentiate nested functions. And why this is literally the same thing as backpropagation in neural networks.

**Level 4 — Partial Derivatives**: When your function has multiple inputs. This is how gradient descent works in practice — ∂Loss/∂w for every single weight.

**Level 5 — Integrals**: Accumulation. Area under the curve. And Monte Carlo integration, which is how you estimate integrals you can't solve analytically (used everywhere in ML).

---

## The Practical Stuff

At the end of the notes there are problems and full solutions. Things like:

- Gradient descent finding a minimum from scratch
- 2D gradient descent with contour plot visualization
- Estimating π with Monte Carlo (throwing random darts, basically)
- Full backpropagation implemented by hand on the XOR problem — no PyTorch, just numpy and chain rule

The XOR one is the one I am most proud of honestly. It's the classic problem that broke single-layer networks in the 60s and caused an "AI winter". We solve it in 50 lines. Just chain rule.

---

## What I Realized

Backpropagation is just the chain rule. Gradient descent is just derivatives. Loss landscapes are just 3D plots of a function with many variables. The "magic" of deep learning is mostly just calculus you learned in high school, applied at scale with good engineering.

That does not make it less impressive. It makes it more impressive. Billions of parameters, all updating via ∂Loss/∂w. Same rule. Every time.

---

No TensorFlow. No sklearn. Just numpy and actually knowing what's happening.

For anyone who wants to learn or just stalk me on GitHub haha. Tschüssi.
