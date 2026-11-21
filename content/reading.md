---
title: Reading list
description: Papers I've been reading or keep coming back to, grouped by topic, with a line on why each one is worth your time.
---

Papers I've been reading or keep coming back to, grouped by topic, with a line on why each one matters. Within each section the first few are the ones I'd start with. Where I wrote a post that builds on a paper, it's linked.

<nav class="reading-toc" aria-label="Sections">

**Sections.** [Start here](#start) · [Reasoning and test-time compute](#reasoning) · [Pruning and stopping](#pruning) · [Inference and the KV cache](#inference) · [Quantization and compression](#quantization) · [Beyond attention](#architectures) · [Vision-language models](#vlm) · [JEPAs and world models](#jepa) · [System One models (Jev)](#jev) · [Interpretability](#interp) · [Alignment and learning from feedback](#alignment) · [Control, sandbagging and elicitation](#control) · [Agents](#agents) · [Scaling and training](#scaling)

</nav>

<h2 id="start">Start here</h2>

If you only read five things on this page, I'd pick these.

- **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)**. Vaswani et al. (2017). The Transformer. Everything else on this list either builds on it or tries to replace it.
- **[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)**. Kaplan et al. (2020). Why people believed that bigger models trained on more data would keep getting better, and roughly how fast.
- **[Toy Models of Superposition](https://arxiv.org/abs/2209.10652)**. Elhage et al. (2022). How a network can store more features than it has neurons, and why that makes interpretability hard.
- **[AI Control: Improving Safety Despite Intentional Subversion](https://arxiv.org/abs/2312.06942)**. Greenblatt, Shlegeris, Sachan and Roger (2024). Assume the model might be trying to cause harm and ask whether your deployment setup still catches it.
- **[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)**. Gu and Dao (2023). One of the most serious challengers to attention, and a good way into state space models.

<h2 id="reasoning">Reasoning and test-time compute</h2>

- **[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)**. Wei et al. (2022). Asking a model to write out its steps makes it much better at multi-step problems.
- **[Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)**. Wang et al. (2022). Sample many reasoning paths and take a majority vote. Simple and still a strong baseline.
- **[Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)**. Lightman et al. (2023). Rewarding each step of a solution beats rewarding only the final answer, and it's where the MATH-500 subset comes from.
- **[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)**. DeepSeek-AI (2025). Long reasoning learned mostly through RL with checkable rewards, plus the small distilled models I use in my experiments.
- **[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)**. Shao et al. (2024). Introduced GRPO, the RL algorithm behind a lot of recent reasoning models. I picked it apart in [Is GRPO Actually a Good Capability Elicitation Algorithm?](/writing/is-grpo-a-good-elicitation-algorithm/).
- **[Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314)**. Snell et al. (2024). When spending more compute at inference beats using a bigger model.
- **[s1: Simple Test-Time Scaling](https://arxiv.org/abs/2501.19393)**. Muennighoff et al. (2025). Controls how long a model thinks by cutting it off or appending "Wait", which is a neat counterpart to the stopping papers below.
- **[Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/abs/2407.21787)**. Brown et al. (2024). How the chance of solving a problem grows with the number of attempts. I used it in [Best-of-N Is an Underrated Baseline](/writing/best-of-n-vs-rl/).
- **[Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://arxiv.org/abs/2504.13837)**. Yue et al. (2025). With enough samples the base model solves more problems than the RL-trained one, which raises the question of what RL actually adds.
- **[Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs](https://arxiv.org/abs/2412.21187)**. Chen et al. (2024). Reasoning models spend far too many tokens on easy problems.
- **[STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465)**. Zelikman, Wu, Mu and Goodman (2022). Train on the model's own correct reasoning and repeat. An early version of the idea behind a lot of current training.

<h2 id="pruning">Pruning, looping and knowing when to stop</h2>

- **[To Reason but Not to Stop: Understanding and Recovering Pruned Large Reasoning Models](https://arxiv.org/abs/2610.19934)**. Nguyen, Ha Lan, Fu, Le and Lu (2026). Argues that a lot of what pruning breaks in reasoning models is stopping, not reasoning, and steers the model toward wrapping up. I tried to replicate it in [Telling a Pruned Reasoning Model When to Stop](/writing/steering-a-pruned-model-to-stop/).
- **[Wait, Wait, Wait... Why Do Reasoning Models Loop?](https://arxiv.org/abs/2512.12895)**. Pipis, Garg, Kontonis, Shrivastava, Krishnamurthy and Papailiopoulos (2025). Traces looping to errors in what the model learned. Raising the temperature cuts the loops but doesn't fix the cause.
- **[Reasoning Models Can Be Accurately Pruned Via Chain-of-Thought Reconstruction](https://arxiv.org/abs/2509.12464)**. Lucas et al. (2025). Pruned reasoning models can end up slower because they think for longer. Calibrating the pruning on the model's own reasoning traces (RAC) fixes a lot of it.
- **[SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot](https://arxiv.org/abs/2301.00774)**. Frantar and Alistarh (2023). One-shot pruning to 50% sparsity with little loss. RAC builds on it, and it is the pruning method I used in my TGAS replication.
- **[A Simple and Effective Pruning Approach for Large Language Models](https://arxiv.org/abs/2306.11695)**. Sun, Liu, Bair and Kolter (2023). Wanda. Prune by weight size times input activation size, with no weight updates, and it works surprisingly well.
- **[LLM-Pruner: On the Structural Pruning of Large Language Models](https://arxiv.org/abs/2305.11627)**. Ma, Fang and Wang (2023). Removes whole structures instead of single weights, so the speedup is real on ordinary hardware.
- **[The Unreasonable Ineffectiveness of the Deeper Layers](https://arxiv.org/abs/2403.17887)**. Gromov et al. (2024). You can drop a big chunk of an LLM's later layers with surprisingly little damage on many tasks.
- **[The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635)**. Frankle and Carbin (2018). Dense networks contain small subnetworks that train just as well on their own. The classic pruning paper.
- **[Deep Compression](https://arxiv.org/abs/1510.00149)**. Han, Mao and Dally (2015). Pruning, quantization and Huffman coding stacked together, from before LLMs.

<h2 id="inference">Inference and the KV cache</h2>

- **[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)**. Kwon et al. (2023). vLLM. Manages the KV cache like virtual memory pages, which is why modern serving fits so many requests at once.
- **[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)**. Dao et al. (2022). Attention is limited by memory traffic, not arithmetic, so reorganize it to stay in fast on-chip memory.
- **[FlashAttention-2](https://arxiv.org/abs/2307.08691)**. Dao (2023). Better work partitioning on the GPU for roughly another 2x.
- **[Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)**. Pope et al. (2022). The clearest explanation I know of why decoding is memory-bound and how to split a model across chips. Pairs well with my [inference post](/writing/the-speed-limit-on-decoding/).
- **[Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)**. Shazeer (2019). Multi-query attention. Share keys and values across heads to shrink the KV cache.
- **[GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)**. Ainslie et al. (2023). The middle ground between one shared head and full multi-head attention that most open models now use.
- **[DeepSeek-V2](https://arxiv.org/abs/2405.04434)**. DeepSeek-AI (2024). Multi-head latent attention compresses the KV cache into a small latent vector.
- **[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)**. Leviathan, Kalman and Matias (2022). A small model drafts tokens and the big model checks them in parallel, with exactly the same output distribution.
- **[Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)**. Chen et al. (2023). The same idea, developed independently at DeepMind.
- **[Medusa](https://arxiv.org/abs/2401.10774)**. Cai et al. (2024). Extra decoding heads on the model itself draft several tokens ahead, so no separate draft model is needed.
- **[EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)**. Li et al. (2024). Drafts at the level of hidden features instead of tokens and gets better acceptance rates.
- **[H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048)**. Zhang et al. (2023). A few tokens get most of the attention, so keep those in the cache and evict the rest.
- **[Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)**. Xiao et al. (2023). Models dump attention on the first few tokens, and keeping them around lets a sliding-window cache run indefinitely.
- **[SnapKV: LLM Knows What You are Looking for Before Generation](https://arxiv.org/abs/2404.14469)**. Li et al. (2024). Picks which parts of a long prompt to keep in the cache by looking at what the end of the prompt attends to.
- **[KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750)**. Liu et al. (2024). Quantize keys per channel and values per token, down to 2 bits.

<h2 id="quantization">Quantization and compression</h2>

- **[LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)**. Dettmers et al. (2022). A few outlier features break naive 8-bit quantization, so handle them separately.
- **[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)**. Frantar et al. (2022). Four-bit weights in one shot, from the same group as SparseGPT and using the same machinery.
- **[SmoothQuant](https://arxiv.org/abs/2211.10438)**. Xiao et al. (2022). Moves the difficulty of activation outliers into the weights so both can be quantized to 8 bits.
- **[AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)**. Lin et al. (2023). Protect the small fraction of weights that matter most, judged by activation size.
- **[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)**. Dettmers et al. (2023). Fine-tune adapters on top of a 4-bit model, which is how a lot of people fine-tune on one GPU.
- **[Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)**. Hinton, Vinyals and Dean (2015). Train a small model on a big model's soft outputs. The origin of distilled models like the ones I prune.

<h2 id="architectures">Beyond attention</h2>

- **[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)**. Gu and Dao (2023). A state space model whose parameters depend on the input, so it can choose what to remember. Linear time in sequence length and no KV cache.
- **[Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060)**. Dao and Gu (2024). Mamba-2. Shows that a class of state space models and a form of linear attention are the same thing, which makes them much faster to train.
- **[Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396)**. Gu, Goel and Ré (2021). S4, the paper that made state space models practical and the starting point for Mamba.
- **[Hyena Hierarchy: Towards Larger Convolutional Language Models](https://arxiv.org/abs/2302.10866)**. Poli et al. (2023). Replaces attention with long convolutions and gating.
- **[RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048)**. Peng et al. (2023). An RNN that trains in parallel like a Transformer and runs like an RNN.
- **[Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621)**. Sun et al. (2023). Another way to get parallel training with recurrent inference.
- **[Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887)**. Lieber et al. (2024). Mixes attention and Mamba layers, which seems to be where things are heading.
- **[Switch Transformers](https://arxiv.org/abs/2101.03961)**. Fedus, Zoph and Shazeer (2021). Mixture of experts with one expert per token, so parameter count grows without compute per token growing.
- **[Mixtral of Experts](https://arxiv.org/abs/2401.04088)**. Jiang et al. (2024). An open mixture-of-experts model that made the approach mainstream.
- **[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)**. Su et al. (2021). Rotary position embeddings, used in nearly every open model now.

<h2 id="vlm">Vision-language models</h2>

- **[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)**. Dosovitskiy et al. (2020). The Vision Transformer. Cut an image into patches and treat them like tokens.
- **[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)**. Radford et al. (2021). CLIP. Match images to their captions at huge scale and you get a vision encoder that understands language.
- **[Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)**. Zhai et al. (2023). SigLIP. A simpler loss than CLIP's that a lot of current vision encoders use.
- **[Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)**. Alayrac et al. (2022). Plugs a frozen vision encoder into a frozen language model with cross-attention.
- **[BLIP-2](https://arxiv.org/abs/2301.12597)**. Li et al. (2023). A small query transformer bridges a frozen image encoder and a frozen LLM.
- **[Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)**. Liu et al. (2023). LLaVA. A simple projection from a vision encoder into an LLM, trained on generated instruction data. The template for many open VLMs.
- **[Qwen2-VL](https://arxiv.org/abs/2409.12191)**. Wang et al. (2024). Handles images at their native resolution and extends rotary position embeddings to 2D and video.
- **[Token Merging: Your ViT But Faster](https://arxiv.org/abs/2210.09461)**. Bolya et al. (2022). Merge similar tokens as they go through the network to speed it up without retraining.
- **[An Image is Worth 1/2 Tokens After Layer 2](https://arxiv.org/abs/2403.06764)**. Chen et al. (2024). FastV. VLMs barely attend to image tokens after the first couple of layers, so you can drop half of them.
- **[CoVisco: Codec-Native Vision Encoder with Native Token Compression for Unified Image-Video Understanding](https://arxiv.org/abs/2609.39924)**. Liu et al. (2026). Takes codec-native video and summarizes each segment with a few learnable tokens, so long videos fit in one pass. With 64 frames it gets close to a much heavier encoder using only 400 visual tokens.

<h2 id="jepa">JEPAs and world models</h2>

Joint-embedding predictive architectures learn by predicting the representation of a missing part of the input, instead of the raw pixels or tokens. The bet is that predicting in an abstract space lets a model ignore unpredictable detail and learn what matters, which is the route Yann LeCun has argued leads to world models.

- **[A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf)**. LeCun (2022). The position paper that lays out JEPAs and why they might be a better path to world models than generative prediction. Long, but the first sections are the whole idea.
- **[Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243)**. Assran et al. (2023). I-JEPA. Predict the representations of masked image blocks from a visible context block, with no hand-made augmentations.
- **[Revisiting Feature Prediction for Learning Visual Representations from Video](https://arxiv.org/abs/2404.08471)**. Bardes et al. (2024). V-JEPA. The same recipe on video, learned purely by predicting features of masked regions.
- **[V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985)**. Assran et al. (2025). Pretrained on over a million hours of video, then turned into an action-conditioned world model with under 62 hours of robot video that can pick and place objects by planning toward an image goal.
- **[LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics](https://arxiv.org/abs/2511.08544)**. Balestriero and LeCun (2025). Argues the embeddings should follow an isotropic Gaussian and adds one regularizer to get there, replacing a pile of training tricks.
- **[MC-JEPA: A Joint-Embedding Predictive Architecture for Self-Supervised Learning of Motion and Content Features](https://arxiv.org/abs/2307.12698)**. Bardes, Ponce and LeCun (2023). One encoder learns both optical flow and content features, and each helps the other.
- **[VL-JEPA: Joint Embedding Predictive Architecture for Vision-language](https://arxiv.org/abs/2512.10942)**. Chen et al. (2025). A vision-language model that predicts text embeddings instead of generating tokens, and only decodes to text when needed.
- **[LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures](https://arxiv.org/abs/2509.14252)**. Huang, LeCun and Balestriero (2025). Adds a JEPA objective to language model training and reports gains over the standard objective.
- **[A-JEPA: Joint-Embedding Predictive Architecture Can Listen](https://arxiv.org/abs/2311.15830)**. Fei et al. (2023). The JEPA recipe applied to audio spectrograms.
- **[How JEPA Avoids Noisy Features: The Implicit Bias of Deep Linear Self Distillation Networks](https://arxiv.org/abs/2407.03475)**. Littwin et al. (2024). In solvable linear models, JEPA-style objectives favor influential features while reconstruction favors high-variance ones. A nice theory companion to the empirical papers.
- **[DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning](https://arxiv.org/abs/2411.04983)**. Zhou, Pan, LeCun and Pinto (2024). A world model that predicts future DINOv2 features instead of pixels and plans by matching goal features.
- **[Bootstrap Your Own Latent](https://arxiv.org/abs/2006.07733)**. Grill et al. (2020). BYOL. Learn by predicting one view's representation from another with no negative examples, an important ancestor of JEPAs.
- **[VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/abs/2105.04906)**. Bardes, Ponce and LeCun (2021). Keeps joint-embedding methods from collapsing with simple variance and covariance penalties.
- **[Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)**. He et al. (2021). The reconstruct-the-pixels alternative that JEPAs are usually compared against.
- **[data2vec](https://arxiv.org/abs/2202.03555)**. Baevski et al. (2022). Predict a teacher's latent representations of masked input, with one recipe for speech, vision and text.
- **[DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)**. Oquab et al. (2023). Strong general-purpose self-supervised image features, and the backbone a lot of world-model work builds on.

<h2 id="jev">System One models (Jev)</h2>

Jev is a different thing from a JEPA. It's a model from the startup TypeSafe AI, announced in September 2026, that TypeSafe calls a "System One model" after Kahneman's fast, intuitive thinking. Instead of writing text, it takes some program state and returns a typed decision, like a choice from a fixed set or a score, with a probability attached, all in one parallel pass. TypeSafe says it's trained with a method it calls Reinforcement Learning for Calibrated Decisions (RLCD) but hasn't published a technical report, so most of the speed, cost and calibration numbers are the company's own. The outside papers below are the best way to see how it actually holds up.

- **[Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)**. Almeida (2026, TypeSafe AI blog). The launch post. It explains the idea and the API, but not how RLCD works.
- **[Evaluating and Benchmarking the System One Model Jev](https://arxiv.org/abs/2609.37647)**. Deußer, Ramamurthy, Sparrenberg and Sifa (2026). An independent zero-shot test on 37 datasets. Jev does very well on standard classification and multiple-choice sets, beats Gemma everywhere and Qwen on most, struggles where everyone struggles, and tends to repeat useless actions in text-game environments.
- **[The Hitchhiker's Guide to System 1 Models: Fast Decision-Making Beyond LLMs](https://arxiv.org/abs/2610.33775)**. Roitman (2026). A survey of models that make fast typed decisions in one forward pass, with a taxonomy and how they pair with slower LLMs in a cascade. A good map of the area.
- **[Jev in the Wild: A Data-Driven Analysis of the Jev Model's Functionality, Applications and Ecosystem](https://arxiv.org/abs/2609.30216)**. Ling, Xue and Ye (2026). Looks at over 2,000 public GitHub projects using Jev. Scoring and attribute judgments are the most common uses.
- **[JevVibe: Efficient Classification-Guided Secure Code Generation](https://arxiv.org/abs/2609.34963)**. Rezvani, Behrouzi and Sadeghi (2026). Uses Jev to label the kind of security weakness in code, then feeds that diagnosis to a repair agent. Competitive with a frontier LLM on ranking the right label while being much faster and cheaper.
- **[DICE: Tilting the Odds through Interface Attacks on System One Model](https://arxiv.org/abs/2610.14404)**. Dong, Lu, Guo and Wang (2026). Even if the model itself is untouched, whoever controls how the state is extracted, how the question is phrased or how the output is read can steer the decision. A useful reminder that "calibrated" only covers the model, not the pipeline around it.
- **[On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)**. Guo et al. (2017). Not about Jev, but the background for what "calibrated probabilities" should mean and how networks usually fail at it.

<h2 id="interp">Interpretability</h2>

- **[Zoom In: An Introduction to Circuits](https://distill.pub/2020/circuits/zoom-in/)**. Olah et al. (2020). The case for reverse engineering networks neuron by neuron and circuit by circuit.
- **[A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)**. Elhage et al. (2021). How to think about attention-only transformers as sums of paths through the residual stream.
- **[In-context Learning and Induction Heads](https://arxiv.org/abs/2209.11895)**. Olsson et al. (2022). A specific circuit that copies patterns from earlier in the context, and evidence it drives in-context learning.
- **[Toy Models of Superposition](https://arxiv.org/abs/2209.10652)**. Elhage et al. (2022). Networks pack more features than neurons by overlapping them.
- **[Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html)**. Bricken et al. (2023). Sparse autoencoders pull interpretable features out of a small transformer.
- **[Sparse Autoencoders Find Highly Interpretable Features in Language Models](https://arxiv.org/abs/2309.08600)**. Cunningham et al. (2023). The same idea, developed in parallel.
- **[Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)**. Templeton et al. (2024). Sparse autoencoders on a production model, with features you can steer.
- **[On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)**. Lindsey et al. (2025). Attribution graphs that trace how a model gets from prompt to answer on real examples.
- **[Interpretability in the Wild](https://arxiv.org/abs/2211.00593)**. Wang et al. (2022). A full circuit for one task in GPT-2 small, indirect object identification.
- **[Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)**. Meng et al. (2022). ROME. Finds where a fact is stored and edits it directly.
- **[Discovering Latent Knowledge in Language Models Without Supervision](https://arxiv.org/abs/2212.03827)**. Burns et al. (2022). Finds a direction for "true" in a model's activations without any labels.
- **[Representation Engineering](https://arxiv.org/abs/2310.01405)**. Zou et al. (2023). Read and control high-level concepts like honesty from activations.
- **[Steering Language Models With Activation Engineering](https://arxiv.org/abs/2308.10248)**. Turner et al. (2023). Add a direction to the activations at inference to change behavior.
- **[Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681)**. Rimsky (now Panickssery) et al. (2024). Difference-of-means steering vectors, the idea TGAS applies to stopping.
- **[Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)**. Arditi et al. (2024). Remove one direction and the model stops refusing. A striking example of how linear some behaviors are.

<h2 id="alignment">Alignment and learning from feedback</h2>

- **[Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565)**. Amodei et al. (2016). An early, very readable list of practical safety problems, most still open.
- **[Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741)**. Christiano et al. (2017). Learn a reward model from people comparing pairs of behaviors.
- **[Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)**. Ouyang et al. (2022). InstructGPT, which made RLHF the standard recipe.
- **[Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)**. Bai et al. (2022). Replace much of the human feedback with feedback from a model following written principles.
- **[Direct Preference Optimization](https://arxiv.org/abs/2305.18290)**. Rafailov et al. (2023). Skip the reward model and train on preferences directly.
- **[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)**. Schulman et al. (2017). The RL algorithm under classic RLHF.
- **[Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)**. Gao, Schulman and Hilton (2023). Push too hard on a learned reward and true quality goes down, with predictable curves.
- **[Weak-to-Strong Generalization](https://arxiv.org/abs/2312.09390)**. Burns et al. (2023). Can a weak supervisor train a stronger model past its own level? Related to my post on [weak labels and sandbaggers](/writing/weak-labels-unlock-then-copy/).
- **[Risks from Learned Optimization in Advanced Machine Learning Systems](https://arxiv.org/abs/1906.01820)**. Hubinger et al. (2019). Where the idea of a model pursuing its own learned objective comes from.
- **[Goal Misgeneralization in Deep Reinforcement Learning](https://arxiv.org/abs/2105.14111)**. Langosco et al. (2021). Agents that stay capable out of distribution but pursue the wrong goal.
- **[Goal Misgeneralization: Why Correct Specifications Aren't Enough For Correct Goals](https://arxiv.org/abs/2210.01790)**. Shah et al. (2022). More examples, and why a correct reward isn't enough.
- **[Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](https://arxiv.org/abs/2401.05566)**. Hubinger et al. (2024). Backdoored behavior that survives standard safety training.
- **[Alignment Faking in Large Language Models](https://arxiv.org/abs/2412.14093)**. Greenblatt et al. (2024). A model that behaves differently when it thinks it's being trained.
- **[Frontier Models are Capable of In-context Scheming](https://arxiv.org/abs/2412.04984)**. Meinke et al. (2024). Models given a goal and a reason to hide it sometimes do.
- **[Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation](https://arxiv.org/abs/2503.11926)**. Baker et al. (2025). Watching chain of thought catches reward hacking, but training against the monitor teaches the model to hide it.
- **[Chain of Thought Monitorability](https://arxiv.org/abs/2507.11473)**. Korbak et al. (2025). Why being able to read a model's reasoning is a fragile opportunity worth protecting.
- **[AI Safety via Debate](https://arxiv.org/abs/1805.00899)**. Irving, Christiano and Amodei (2018). Have two models argue and let a human judge, so the judge can supervise more than they could alone.

<h2 id="control">Control, sandbagging and elicitation</h2>

- **[AI Control: Improving Safety Despite Intentional Subversion](https://arxiv.org/abs/2312.06942)**. Greenblatt, Shlegeris, Sachan and Roger (2024). The paper that set up control evaluations. Several of my posts, like [Why a Failed Red-Team Evaluation Is Not Evidence of Safety](/writing/failed-red-team-evaluation/), build on it.
- **[Ctrl-Z: Controlling AI Agents via Resampling](https://arxiv.org/abs/2504.10374)**. Bhatt et al. (2025). Control for multi-step agents, where suspicious actions get resampled.
- **[Games for AI Control](https://arxiv.org/abs/2409.07985)**. Griffin, Thomson, Shlegeris and Abate (2024). Treats control evaluations as games and solves for the best protocols.
- **[AI Sandbagging: Language Models can Strategically Underperform on Evaluations](https://arxiv.org/abs/2406.07358)**. van der Weij et al. (2025). Models can be prompted or trained to underperform on purpose, even selectively.
- **[Stress-Testing Capability Elicitation With Password-Locked Models](https://arxiv.org/abs/2405.19550)**. Greenblatt, Roger, Krasheninnikov and Krueger (2024). Models that hide a skill unless given a password, and which training methods can draw it back out.
- **[Exploration Hacking: Can LLMs Learn to Resist RL Training?](https://proceedings.mlr.press/v306/jang26b.html)**. Jang et al. (2026). Whether a model can avoid being trained into a capability by never trying it. My [first post](/writing/zero-reward-doesnt-mean-zero-capability/) is the toy version.
- **[Noise Injection Reveals Hidden Capabilities of Sandbagging Language Models](https://arxiv.org/abs/2412.01784)**. Tice et al. (2025). Add noise to the weights and see whether the score goes up. I looked at when that works in [Noise Injection Only Catches Fragile Sandbaggers](/writing/noise-injection-fragile-sandbaggers/).
- **[Auditing Games for Sandbagging](https://arxiv.org/abs/2512.07810)**. Taylor et al. (2025). A blue team tries to catch real sandbagging models built by a red team, and most detection methods struggle.

<h2 id="agents">Agents</h2>

- **[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)**. Yao et al. (2022). Interleave reasoning with tool calls. The basic loop behind most agents.
- **[Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)**. Schick et al. (2023). A model learns when to call APIs from its own filtered attempts.
- **[Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)**. Park et al. (2023). A little town of LLM agents with memory and plans.
- **[SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770)**. Jimenez et al. (2023). The benchmark that made coding agents measurable.
- **[Measuring AI Ability to Complete Long Tasks](https://arxiv.org/abs/2503.14499)**. Kwa et al. (2025). METR's time-horizon measure, the length of task a model can finish half the time, and how fast it's growing.
- **[Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325)**. Du et al. (2023). Several copies of a model debate their answers and get more accurate.
- **[Towards a Science of Scaling Agent Systems](https://arxiv.org/abs/2512.08296)**. Kim et al. (2025). A big controlled comparison of single-agent and multi-agent setups. More agents help less once a single agent is already strong, and designs without a central checker let errors spread.
- **[A Case Study on Emergent Cheating and Whistleblowing in Autonomous Research Swarms](https://arxiv.org/abs/2609.04170)**. Paglieri, Cross, Genewein, Leibo, Tomasev and Vezhnevets (2026, Google DeepMind). A hundred agents working on formal proofs. One found an exploit and it spread, while another group audited the fake proofs and flagged them.

<h2 id="scaling">Scaling and training</h2>

- **[Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)**. Brown et al. (2020). GPT-3 and the discovery that large models can learn tasks from a few examples in the prompt.
- **[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)**. Kaplan et al. (2020). Loss falls as a power law in parameters, data and compute.
- **[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)**. Hoffmann et al. (2022). Chinchilla. Most models were too big for their data, and you should scale both together.
- **[Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682)**. Wei et al. (2022). Abilities that seem to appear suddenly at scale.
- **[Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004)**. Schaeffer, Miranda and Koyejo (2023). Much of that suddenness comes from the choice of metric. I leaned on this in [How Much Can a Sandbagger Hide Under a Scaling Trend?](/writing/hiding-under-the-scaling-trend/).
- **[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)**. Hu et al. (2021). Fine-tune small low-rank updates instead of all the weights.
- **[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)**. Kingma and Ba (2014). The optimizer behind almost everything here, and the reason several of my toy results look the way they do.
- **[Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177)**. Power et al. (2022). Networks that suddenly generalize long after they've memorized the training set.
- **[Deep Double Descent: Where Bigger Models and More Data Hurt](https://arxiv.org/abs/1912.02292)**. Nakkiran et al. (2019). Test error can get worse and then better again as models grow.
