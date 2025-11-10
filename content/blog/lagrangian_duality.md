+++
title = "Lagrangian Duality, KKT conditions and LP duality"
date = "2025-11-09T13:44:57+01:00"

#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = ["Math", "Optimization", "KKT", "Convex optimization", "Linear Programming"]
+++

## Optimization context
Let's imagine for a moment we are trying to solve a class of *constrained* optimization problems of the form
$$\begin{align*}
    &\inf_x f(x) \\
    &\text{subject to } &g_i(x) \leq 0 \text{ for } 1 \leq i \leq m \\
                        &&h_j(x) = 0, \text{ for } 1 \leq j \leq n
\end{align*} 
$$
where $f, g_1, \ldots, g_m, h_1, \ldots, h_n \in \mathcal{C}$ for $\mathcal{C}$ a fixed class of smooth functions. We fix such a problem $(P)$. How should we do it? 


<!---
Note: This framework also encompasses *equality* constraints $h(x) = 0$ if whenever $f \in \mathcal{C}$ we also have $-f \in \mathcal{C}$. Indeed, one can introduce the two constraints $h(x) \leq 0 \land h(x) \geq 0$, amounting to $h(x) = 0$. It turns out this 
-->


It is mathematical common sense that unconstrained optimization problems are simpler to solve than their constrained counterparts. So we may be tempted to transform the problem into
$$\inf_x f(x) + \sum_{i = 1}^m J(g_i(x)) + \sum_{j = 1}^n I(h_j(x))$$

where 
$$J(x) = \begin{cases} 0 \text{ if } x \leq 0 \\ +\infty \text{ otherwise} \end{cases} \text{ and } I(x) = \begin{cases} 0 \text{ if } x = 0 \\ +\infty \text{ otherwise} \end{cases} $$ 

Unfortunately, $I$ and $J$ are not smooth—indeed, they are not even continuous—so smooth optimization methods cannot be applied, leaving us at a dead end. Something a little smarter would be to replace $I$ and $J$ by $J_{\lambda_i}, I_{\mu_j}$, where 
$$J_{\lambda_i} (x) = \begin{cases} 0 \text{ if } x \leq 0 \\ \lambda_i \cdot x \text{ otherwise} \end{cases} \text{ and } I_{\mu_j} (x) = \begin{cases} 0 \text{ if } x = 0 \\ \mu_j \cdot x \text{ otherwise} \end{cases}$$ 
and 
solve 
$$\inf_x f(x) + \sum_{i = 1}^m J_{\lambda_i} (g_i(x)) + \sum_{j = 1}^n I_{\mu_j} (h_j(x)) \leq 0$$

with $\lambda, \mu$ as parameters and then take $\lambda, \mu \to +\infty$. But is this the same thing as solving $(P)$? It turns out that the answer is not too complicated, and that this is not a dead end at all. The standard theory avoids $I_{\mu_j}$ and $J_{\lambda_i}$ and instead simply uses multiplication by scalars, which is what we will do too as it is arguably simpler. The function

$$\mathcal{L}(x, \lambda, \mu) = f(x) + \sum_{i = 1}^m \lambda_i \cdot g_i(x) + \sum_{j = 1}^n \mu_j \cdot h_j(x)$$
is called the Lagrangian of the optimization problem (P). Letting the domain of the optimization problem be $K = \{x \mid \forall i, j, g_i(x) \leq 0, h_j(x) = 0\}$, we have
> The Lagrangian embeds all the data of $(P)$, in that:
> - For any fixed parameter $x$, we have 
> $$\sup_{\lambda \geq 0, \mu} \mathcal{L}(x, \lambda, \mu) = \begin{cases} f(x) \text{ if } x \in K \\ + \infty \text{ otherwise} \end{cases}$$
> - It follows that $(P)$ is equivalent to solving
>  $$\inf_x \sup_{\lambda \geq 0, \mu} \mathcal{L}(x, \lambda, \mu)$$

But how does this help us? We began with one optimization problem, and now seem to have two nested ones. Well, the advantage is that these new problems are unconstrained! Also, suppose for a moment that we were able to switch the $\inf$ and the $\sup$ in the last equation. Then we would be solving for
$$\sup_{\lambda \geq 0, \mu} \inf_x \mathcal{L}(x, \lambda, \mu)$$
The function $\inf_x \mathcal{L}(x, \cdot, \cdot)$ is a pointwise infimum of affine maps, meaning it is concave: the supremum problem becomes easy to solve. We are only left with a single (parametrized) infimization unconstrained problem, which is arguably a lot better than what we started with. 

We made a crucial assumption, let's go back to it:
$$ \inf_x \sup_{\lambda \geq 0, \mu} \mathcal{L}(x, \lambda, \mu) =^{???!}\sup_{\lambda \geq 0, \mu} \inf_x \mathcal{L}(x, \lambda, \mu) $$

This second problem is called the dual problem of $(P)$.
> Given a problem $(P)$, we say that the problem 
> $$ (D) \  \ \sup_{\lambda \geq 0, \mu} \inf_x \mathcal{L}(x, \lambda, \mu) $$
> is the dual of $(P)$.

Denote by $p^*$ and $d^*$ the optimal value of $(P)$ and $(D)$ respectively. Since
$$\inf_x \sup_{\lambda \geq 0, \mu} \mathcal{L}(x, \lambda, \mu) = p^*$$
we have, for given $\lambda \geq 0$ and $\mu$,
$$\inf_x \mathcal{L}(x, \lambda, \mu) \leq p^*$$
and thus 
$$\sup_{\lambda \geq 0} \inf_x \mathcal{L}(x, \lambda, \mu) = d^* \leq p^*$$

The inequality $d^* \leq p^*$ is known as *weak* Lagrangian duality and it is often useful to get lower bounds on certain classes of optimization problems. 
The reverse inequality can be false, and we say *strong* Lagrangian duality holds when it is true, i.e. $p^* = d^*$. The difference $p^* - d^*$ is called the *duality gap*.

{{< details title="Click for counter-example to strong duality">}}
Take for instance the convex problem (!)
$$\begin{align*}
    &\inf_{x, y > 0} e^{-x} \\
    &\text{subject to } \frac{x^2}{y} \leq 0
\end{align*}$$
It is obvious that $p^* = 1$, and $\mathcal{L}(x, y, \lambda) = e^{-x} + \lambda \frac{x^2}{y}$ whose infimum on $x, y > 0$ is $0$, for any $\lambda \geq 0$, i.e. $d^* = 0$

{{< /details >}}



## Slater's condition for strong duality

As we just saw, convexity is not enough to guarantee strong duality. It turns out it is not too far from being enough:
> __Slater's condition__:
> 
> If the problem is smooth convex, i.e. $f, g_1, \ldots, g_m$ are all smooth convex and $h_1, \ldots, h_n$ are all affine, and strictly feasible, i.e. there exists $x$ such that $g_i(x) < 0$ for all inequality constraints $g_i$ that are not affine functions, then strong duality holds.
> 
> Moreover, the dual value is reached, i.e. there exists $(x^*, \lambda^*)$ solution of the dual problem.

{{< details title="About the smoothness condition">}}

One can actually remove the smoothness condition completely, and work with subgradients instead. While sometimes useful, the generalization is confusing if the reader does not know about subgradients and straightforward otherwise. I choose not to dive into that. 

{{< /details >}}

The proof is highly geometric and full details can be found in [Convex Optimization, by Boyd and Vandenberghe](https://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf), at the end of page 234. The drawings and motivation in the book is very nice, and the following proof is only a partial, concise, reconstruction of these.


{{< details title="The reconstruction">}}

We follow notations of the above book, which are not exactly compatible with the notations of this post: they write the affine equality condition as $Ax = b$ with $A$ a matrix of full rank $p$ (which has nothing to do with $p^*$, although they use it the same way we do...). The letters $\lambda, \mu$ are not associated with scalars for (in)equality constraints respectively anymore, and they use $\lambda, \nu$ instead. 

Feasibility ensures $p^* \neq +\infty$, and if $p^* = -\infty$, weak duality gives $d^* = -\infty$. Let $p^*$ be finite, and define 
$$\begin{align*}
&\mathcal{A} = \{(u,v,t) | \exists x, g_i(x) \leq u_i, (Ax)_i = b + v_i, f(x) \leq t \} \\
&\mathcal{B} = \{(0, 0, s)| s < p^* \}
\end{align*}$$
which can be thought of as some kind of epigraph of the "reachable values" by the problem. 
Both of these are convex and do not intersect, so the [separating hyperplane theorem](https://en.wikipedia.org/wiki/Hyperplane_separation_theorem) gives some $(\tilde{\lambda}, \tilde{\nu}, \mu)$ and $\alpha$ so that the linear form 
$$\varphi(u, v, t) = (\tilde{\lambda}, \tilde{\nu}, \mu)^T (u, v, t) + \alpha$$ 
separates $\mathcal{A}$ and $\mathcal{B}$, i.e. $\mathcal{A} \subset \{\varphi \geq \alpha\}$ and $\mathcal{B} \subset \{\varphi \leq \alpha\}$.

Since $\mathcal{A}$ is unbounded above, we must have $\tilde{\lambda} \succeq 0, \mu \geq 0$. 
Condition for $\mathcal{B}$ gives $\mu t \leq \alpha$ for any $t < p^*$, i.e. $\mu p^* \leq \alpha$. 

If $\mu > 0$, by taking $\lambda = \frac{\tilde{\lambda}}{\mu}, \nu = \frac{\tilde{\nu}}{\mu}$ we get that $\mathcal{L}(x, \lambda, \nu) \geq p^*$, which gives by taking minimum over $x$ that $d^* \geq p^*$, i.e. $d^* = p^*$ by weak duality, and $(\lambda, \mu)$ is an optimal dual point.

The case $\mu = 0$ cannot happen. Indeed, by writing the condition for $\mathcal{A}$ at the point $x^0$ satisfying Slater's condition, we have 

$$\sum_{i = 1}^m \tilde{\lambda}_i f_i(x^0) + \nu^T (Ax^0 - b) \geq \alpha \geq \mu p^* = 0$$
that $\tilde{\lambda} = 0$ since $f_i(x^0) < 0$.

As $(\tilde{\lambda}, \tilde{\nu}, \mu) \neq 0$, we must have $\tilde{\nu} \neq 0$. From $v^T(Ax^0 - b) \geq 0$, and $x^0$ being in the interior of the domain of definition, we must have some point $x$ such that $v^T(Ax - b) \leq 0$, unless $v^T A = 0$, contradicting $\text{rank } A = p$.

{{< /details >}}

## The Karush-Kuhn-Tucker conditions

Strong Lagrangian duality allows us to derive general necessary conditions for a point $x^*$ to be an optimum (at least whenever the duality gap is zero). We go back, for the moment, to the general where $\mathcal{C}$ is any class of smooth functions.

> __KKT necessary conditions__ (smooth case)
>
> Let $x^*$ be an optimal point for a smooth $(P)$ and assume that the duality gap for $(P)$ is zero. Then the KKT conditions state that there exists $\lambda^*, \mu^*$ such that:
> - The gradient of the Lagrangian w.r.t $x$ is zero at $(x^*, \lambda^*, \mu^*)$:
> $$ \nabla f(x^*) + \sum_{i = 1}^m \lambda_i^* \nabla g_i(x^*) + \sum_{j = 1}^n \mu_j^* \nabla h_j(x) = 0 $$
> - $\lambda^* \succeq 0$ and $\lambda_i^* g_i(x^*) = 0$ for each $1 \leq i \leq m$

Pick $(\lambda^*, \mu^*)$ a dual optimal point. The gradient vanishing is given by the fact $x^*$ solves $(P)$, $\lambda^* \succeq 0$ follows from constraints and $\lambda_i^* g_i(x^*)$ follows from $f(x^*) = \mathcal{L}(x^*, \lambda^*, \mu^*)$ and the combination of $\lambda^*_i \leq 0$ and $g_i(x^*) \leq 0$.

Note that in the unconstrained case, the conditions reduce to being a critical point.


> __KKT conditions__ (convex case)
>
> If $(P)$ is assumed to be smooth convex (i.e. $f, g_1, \ldots, g_m$ are smooth convex maps and $h_j$ are affine), the necessary conditions are sufficient: 
>  
> For any primal feasible point $x^*$, if there exists $(\lambda^*, \mu^*)$ satisfying the two points above then $x^*$ is primal optimal and $(\lambda^*, \mu^*)$ is dual optimal. Moreover the duality gap is automatically zero.

Since $\lambda^* \succeq 0$, the map $x \mapsto \mathcal{L}(\lambda^*, \mu^*, x)$ is convex and the first KKT condition shows that its gradient vanishes at $x^*$, i.e. it is a minimum. Then $\mathcal{L}(\lambda^*, \mu^*, x^*) = f(x^*)$ as $x^*$ is feasible and satisfies the second KKT condition. This equality enforces that the duality gap is zero and in particular that $x^*$ and $(\lambda^*, \mu^*)$ are primal and dual optimal points respectively.

## About LP duality

This post ends here, as I believe we got a reasonable answer to the question at the beginning: whenever $\mathcal{C}$ is a class of smooth (or not, reformulating everything with subgradients) convex maps, we found a necessary and sufficient effective condition for optimality. 

It is however kind of sad to develop all this theory without saying even a single word about how it can be used. Problems of the form
$$
\begin{align*}
    &\inf_x c^T x \\
    &\text{subject to } Ax \preceq  b
\end{align*} 
$$
are called linear programs (LP), and it is standard LP theory that the above program has a dual equivalent program that's given by

$$
\begin{align*}
    &\sup_y b^T y \\
    &\text{subject to } A^Ty = c, y \succeq 0
\end{align*} 
$$

and that whenever either one has a solution, the other also has one and the objectives are equal. It’s a nice exercise to use the framework developed here to prove it.

This implies that finding a feasible point for a general LP is as hard as finding an optimal one. Can you see why?