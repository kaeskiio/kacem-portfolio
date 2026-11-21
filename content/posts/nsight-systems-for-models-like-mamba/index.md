---
title: What Nsight Systems Shows You, and How I'd Point It at Mamba
summary: What NVIDIA's Nsight Systems timeline offers, how it fits next to Nsight Compute, and how I would use it to find launch gaps and hidden syncs in a model like Mamba-2.
category: Inference
tags: Nsight Systems, profiling, Mamba-2
order: 3.0
---

*A profiler that draws a timeline of everything your CPU and GPU did is the first tool I'd reach for on a model like Mamba-2, where the hard part is often the gaps between kernels and not the kernels themselves.*

My last inference post ended with a number, a [decode speed limit](/writing/the-speed-limit-on-decoding/) set by memory bandwidth. Getting close to a number like that means finding out where the time actually goes, and on a GPU that's surprisingly hard to guess. This post is my take on NVIDIA's Nsight Systems, what it offers, how it fits next to the other profilers, and how I'd use it on a state space model like [Mamba-2](/writing/thoughts-on-mamba2/). I haven't run this on Mamba yet, so treat the Mamba parts as a plan, not results.

## What it is

Nsight Systems records what your program did over a stretch of time and draws it as a timeline. You see rows for CPU threads, the CUDA API calls they made, the kernels and memory copies that ran on the GPU, and any labeled ranges you added to your own code. It can also sample GPU-wide metrics over time, like memory bandwidth and how busy the streaming multiprocessors are. According to NVIDIA's docs it needs no changes to your program to work at all, though a few labels make it far more useful.

The big thing it shows that other tools don't is the relationship between CPU and GPU. A kernel can be perfectly optimized and your model can still be slow because the GPU sat idle waiting for Python to launch the next kernel, or because some line quietly forced a synchronization. Nsight Systems makes those gaps visible as literal empty space on the GPU row.

## Where it fits

I think of three tools as zoom levels.

- **Nsight Systems** is the wide shot. What ran, in what order, for how long, and where the GPU was idle. It's the place to start.
- **Nsight Compute** is the close-up on one kernel. It replays the kernel and collects detailed hardware counters, so you can see whether it's limited by memory bandwidth, compute or something else. The intended workflow is to find the important kernel on the timeline first, and NVIDIA's docs describe right-clicking a kernel launch on the timeline and choosing Profile Kernel to hand it to Nsight Compute.
- **The PyTorch profiler** is in between and speaks PyTorch, with operator names and shapes. It's convenient, but it doesn't show the whole system as directly.

So my rule is to never optimize a kernel before looking at the timeline. If half the time is launch gaps, a faster kernel barely matters.

## Why Mamba is a good target

Mamba-2 has two very different modes, and each one has a different failure you'd want to catch.

**Prefill (and training) runs the chunked algorithm.** In the official code each Mamba-2 layer launches, in order, Triton kernels named `_chunk_cumsum_fwd_kernel`, `_chunk_state_fwd_kernel`, `_state_passing_fwd_kernel`, `_bmm_chunk_fwd_kernel` and `_chunk_scan_fwd_kernel`, plus the causal convolution, a gated RMSNorm and the big input and output projections. The questions I'd want answered are simple. How much of a layer's time is the regular matrix multiplies versus the five scan kernels? Is the state-passing step, the only sequential part, ever on the critical path? Is the GPU busy the whole time, or are there gaps between these launches? On the timeline those answers are just bars and spaces you can measure.

**Decode is a pile of tiny kernels.** Generating one token runs, per layer, a small matrix-vector product for the input projection, a convolution update, a fused state update (`_selective_scan_update_kernel` in the Triton version), a gated norm and the output projection. At batch size 1 the projections are memory-bound matrix-vector products and the convolution and state updates are tiny, so it's easy for the CPU to take longer to launch them all than the GPU takes to run them. The official generation code has an option to capture decoding in a CUDA graph for exactly this reason, so the whole step launches at once. Nsight Systems is how you'd see whether that's working. Without graphs I'd expect the GPU row to look like a comb, with thin kernels separated by gaps. With graphs it should look solid.

## How I'd actually run it

This is the setup I'd use, based on the Nsight Systems docs.

**Label the phases.** Wrap prefill and each decode step in NVTX ranges, using `torch.cuda.nvtx.range_push("prefill")` and `range_pop()`. Those labels show up on the timeline and let the summary reports group kernels by phase. Nsight Systems also has a `--pytorch=autograd-nvtx` option that adds ranges for PyTorch operations automatically.

**Skip the warmup.** Triton compiles kernels the first time they run, and autotuning tries several configurations. That shows up as long CPU stretches that have nothing to do with steady-state speed. I'd run a few warmup steps first, then call `torch.cuda.profiler.start()` and `stop()` around the part I care about, and profile with `--capture-range=cudaProfilerApi` so only that window is recorded.

**Pick what to trace.** Something like

```
nsys profile -t cuda,nvtx,osrt --capture-range=cudaProfilerApi \
    --cuda-graph-trace=node -o mamba_decode python generate.py
```

The `--cuda-graph-trace=node` flag shows the individual kernels inside a CUDA graph instead of one opaque block. The docs warn that it can add a lot of overhead, so I'd use it to understand the graph and then turn it off to time it. Adding `--gpu-metrics-devices=all` samples bandwidth and utilization over time, which is the closest the timeline gets to telling you how near the memory roofline you are.

**Read the summaries, not just the picture.** `nsys stats` turns a report into tables. The `cuda_gpu_kern_sum` report lists total time per kernel, which answers "scan kernels versus matrix multiplies" directly. `nvtx_kern_sum` groups kernels by the NVTX range they ran in, so prefill and decode come out separately. `nsys analyze` has rules that flag common problems, like `gpu_gaps` for stretches where the GPU sat idle and `cuda_api_sync` for calls that blocked the CPU.

**Then zoom in.** Once the timeline says a particular kernel matters, that's when Nsight Compute is worth the time.

## What I'd expect to find

These are guesses I'd want to check, not results.

- **Decode at batch 1 is launch-bound without CUDA graphs.** The per-token work in each layer is small, so I'd expect visible gaps between kernels, and a big improvement from graph capture.
- **Prefill is dominated by the projections, not the scan.** The input and output projections are large matrix multiplies, and the whole point of the chunked algorithm is to make the scan itself mostly matrix multiplies too. I'd guess the five scan kernels are a minority of layer time at long sequence lengths, but it's exactly the kind of guess the timeline would correct.
- **Hidden syncs.** Things like reading a tensor's value back to Python to check for an end-of-sequence token force the CPU to wait for the GPU. The official generation loop does exactly this check every step, so it's worth looking for. In a decode loop that can quietly cost a lot, and it shows up clearly in the timeline and in the `cuda_api_sync` rule.

## Things to watch out for

- **Profiling changes timing.** Tracing adds overhead, especially with node-level graph tracing, so I'd use one run to understand what's happening and a lighter run to time it.
- **Permissions.** Hardware counters often need extra permissions, and some cloud containers block them. It's worth checking on the first day with a new machine, which is something I learned while planning my own engine.
- **The first run lies.** Compilation, autotuning and memory allocation all happen early. Always profile a steady state.

## What I'd take from this

Nsight Systems is the tool that tells you what to optimize, not how. For a model like Mamba-2, where the clever math already turned most of the work into matrix multiplies, I'd expect the remaining problems to be about orchestration. That means kernel launches, synchronization and how decode steps are batched, and those are exactly what a timeline shows best.

## Things I'm still curious about

- How much does CUDA graph capture actually help Mamba-2 decode at batch 1, compared with a Transformer of the same size?
- At what sequence length does the prefill time shift from the projections to the scan kernels, if it ever does?
- How close does the fused state-update kernel get to the memory bandwidth limit, which is the Nsight Compute question that comes after the timeline?

## References

- NVIDIA. Nsight Systems User Guide. https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- NVIDIA. Nsight Systems Post-Collection Analysis Guide. https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html
- NVIDIA. Nsight Compute Documentation. https://docs.nvidia.com/nsight-compute/NsightCompute/index.html
- Dao, T., and Gu, A. (2024). Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. ICML 2024. arXiv:2405.21060. https://arxiv.org/abs/2405.21060
- Official Mamba code (state-spaces/mamba), including the Triton SSD kernels and the generation utilities with CUDA graph support. https://github.com/state-spaces/mamba
