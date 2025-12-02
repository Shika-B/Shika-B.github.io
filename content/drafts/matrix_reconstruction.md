+++
title = "Matrix reconstruction and denoising"
date = "2025-11-10T02:47:29+01:00"

#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = ["Math", "Vision", "Low rank approximation", "Denoising"]
+++

Consider the following matrix, with some entries unknown:

$$
M = \begin{pmatrix}
? & 2 & ? \\
2 & 4 & ? \\
3 & ? & 3
\end{pmatrix}
$$

What is the full matrix ? That's an absurd question, right ? The only way to get the full matrix is to know all of its entries, after all...
Or is it ? For instance if you also know $\text{rank}M = 1$, then you know actually know that 
$$
M = \begin{pmatrix}
1 & 2 & 1 \\
2 & 4 & 2 \\
3 & 6 & 3
\end{pmatrix}
$$

Generalizing the generalization of the generalization of the above argument gives
> [Candes]

Oh and also, all of this is effective and we can actually compute the completion of $M$ very quickly. Wtf, right ? 