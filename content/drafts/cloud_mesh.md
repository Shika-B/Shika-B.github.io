+++
title = "Poisson surface reconstruction"
date = "2025-11-29T18:08:29+01:00"

#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = ["Computer Science", "Mathematics", "Computer Vision", "Differential Equations"]
+++

A laser scan of a 3D object hands you a set of point and a set of vectors: 
samples sitting on the object's surface and the normals at each of these points.
That's awkward, because almost everything you'd want to do with the object 
(render it, 3D-print it, measure it, drop it in a physics engine) needs an 
actual surface, not a point cloud. So the question is: given only these samples,
how do we recover the surface they came from?

I asked that to a few people and they all imagined the answer to be a complex 
set of heuristics stacked one on top of the other, which is understandable given
under-constrained the problem is. It turns out the answer is both simple and 
mathematically pretty.

## A precise statement

A bounded geometric object $\Omega \subset \mathbb{R}^3$ is watertight if its 
surface $\partial \Omega$ would be able to contain water without spilling any 
of it, no matter how much you play with it[^1]. We consider a scan of that 
surface, that is a family of points $(s_i)_{i \in I}$ and vectors 
$(n_i)_{i \in I}$ with $s_i \in \partial \Omega$ and $n_i$ being the 
outward-pointing normal vector at $s_i$.

## The idea 
On one hand, this scan can be seen as samples of the gradient[^2] of 
$\mathbf{1}_\Omega$. Indeed, the map $\nabla \mathbf{1}_\Omega$ is exactly the 
exterior normal map on $\partial \Omega$, and zero outside. 

On the other hand, the 
[divergence theorem](https://en.wikipedia.org/wiki/Divergence_theorem) gives
$$\int\int\!\!\!\!\int_\Omega \nabla \cdot V\ \mathrm{d}V = 
\int\!\!\!\!\int_{\partial \Omega} V \cdot \hat{n} \ \mathrm{d}\partial\Omega$$

[^1]: Formally, we ask $\partial \Omega$ to be a closed $2$-manifold, i.e. 
compact with no boundary.

[^2]: Either consider we are talking distributionnal-gradient or that 
we smoothed the function by a convolution with a peaky gaussian. 