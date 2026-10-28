---
title: Post title in sentence case
date: 2026-11-01
summary: One sentence that shows up under the title in post lists and link previews.
draft: true
---

Copy this file to `content/posts/your-slug.md` (or `content/posts/your-slug/index.md` if the
post has images) and remove `draft: true` when it's ready. Files starting with `_` are ignored.
Drafts only appear when you run `npm run dev`.

## Headings

Use `##` for sections and `###` for subsections. The title above is the only `#`.

## Math

Inline math goes between single dollar signs, like $\pi_\theta(a_t \mid s_t)$, or between `\(` and `\)`.
Display math goes between double dollar signs or `\[` and `\]`:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat A_t,\ \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat A_t\right)\right]
$$

A dollar amount like $20 or $5 per hour is left alone.

## Figures

Put images in the post's folder and reference them by file name:

<figure>
  <img src="proxy-vs-true.png" alt="Describe what the plot shows">
  <figcaption>Figure 1. Proxy reward keeps rising after true reward peaks.</figcaption>
</figure>

## Code

```python
ratio = torch.exp(logp - logp_old)
loss = -torch.min(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv).mean()
```

## Quotes and footnotes

> Use block quotes for quoting papers, sparingly.

Link to sources inline: [Schulman et al., 2017](https://arxiv.org/abs/1707.06347).
