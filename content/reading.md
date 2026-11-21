---
title: Reading list
description: Papers I've been reading or keep coming back to, with a line on why each one is worth your time.
---

Papers I've been reading or keep coming back to, with a line or two on why. Some of them have a post of mine attached where I tried something related.

## Pruning, looping and knowing when to stop

- **[To Reason but Not to Stop: Understanding and Recovering Pruned Large Reasoning Models](https://arxiv.org/abs/2610.19934)**. Minh-Anh Nguyen, Nguyen T. Ha Lan, Xiyan Fu, Dung D. Le and Wei Lu (2026). Argues that a lot of what pruning breaks in reasoning models is stopping, not reasoning, and steers the model toward wrapping up with a direction read off its activations. I tried to replicate it in [Telling a Pruned Reasoning Model When to Stop](/writing/steering-a-pruned-model-to-stop/).
- **[Wait, Wait, Wait... Why Do Reasoning Models Loop?](https://arxiv.org/abs/2512.12895)**. Charilaos Pipis, Shivam Garg, Vasilis Kontonis, Vaishnavi Shrivastava, Akshay Krishnamurthy and Dimitris Papailiopoulos (2025). Looks at why reasoning models get stuck repeating themselves, especially at low temperature, and traces it to errors in what the model learned. Raising the temperature cuts the loops but doesn't fix the cause, so outputs stay longer than they need to be.
- **[Reasoning Models Can Be Accurately Pruned Via Chain-of-Thought Reconstruction](https://arxiv.org/abs/2509.12464)**. Ryan Lucas, Kayhan Behdin, Zhipeng Wang, Qingquan Song, Shao Tang and Rahul Mazumder (2025). Shows that pruned reasoning models can end up slower than the originals because they think for longer, and fixes the pruning by calibrating on the model's own reasoning traces (RAC).
- **[SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot](https://arxiv.org/abs/2301.00774)**. Elias Frantar and Dan Alistarh (2023). The one-shot pruning method that TGAS and RAC both build on.
- **[Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681)**. Nina Rimsky (now Panickssery), Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger and Alexander Matt Turner (2024). The difference-of-means steering idea that TGAS applies to stopping.

## Vision-language models

- **[CoVisco: Codec-Native Vision Encoder with Native Token Compression for Unified Image-Video Understanding](https://arxiv.org/abs/2609.39924)**. Yulong Liu, Xiaotian Han, Junyuan Shang, Yuchen Ding, Zhenyu Zhang, Shuohuan Wang, Guibo Zhu, Sirui Han and Dianhai Yu (2026). A vision encoder that takes codec-native video and splits a long video into segments, each summarized by a few learnable tokens, so it can encode many frames in one pass. With four segments and 64 frames it gets close to a much heavier encoder using only 400 visual tokens.

## Agents working together

- **[A Case Study on Emergent Cheating and Whistleblowing in Autonomous Research Swarms](https://arxiv.org/abs/2609.04170)**. Davide Paglieri, Logan Cross, Tim Genewein, Joel Z. Leibo, Nenad Tomasev and Alexander Sasha Vezhnevets (2026, Google DeepMind). A hundred agents working on formal math proofs. One found an exploit, it spread through their shared library and messages, and another group of agents started auditing the fake proofs and flagging them. The same open communication enabled both the cheating and the pushback.
- **[Towards a Science of Scaling Agent Systems](https://arxiv.org/abs/2512.08296)**. Yubin Kim, Ken Gu, Chanwoo Park and 17 others (2025, a team at Google). A big controlled comparison of single-agent and multi-agent setups across benchmarks and model families. Multi-agent setups help less once the single agent is already strong, and designs without a central checker let errors spread.

## AI control and sandbagging

- **[AI Control: Improving Safety Despite Intentional Subversion](https://arxiv.org/abs/2312.06942)**. Ryan Greenblatt, Buck Shlegeris, Kshitij Sachan and Fabien Roger (2024). The paper that set up control evaluations, where a red team plays a scheming model and you measure whether the deployment setup still catches it. Several of my posts, like [Why a Failed Red-Team Evaluation Is Not Evidence of Safety](/writing/failed-red-team-evaluation/), build on it.
- **[Stress-Testing Capability Elicitation With Password-Locked Models](https://arxiv.org/abs/2405.19550)**. Ryan Greenblatt, Fabien Roger, Dmitrii Krasheninnikov and David Krueger (2024). Builds models that hide a skill unless given a password and checks which training methods can draw it back out.
- **[Exploration Hacking: Can LLMs Learn to Resist RL Training?](https://proceedings.mlr.press/v306/jang26b.html)**. Jang et al. (2026). Whether a model can avoid being trained into a capability by never trying it during RL. My [first post](/writing/zero-reward-doesnt-mean-zero-capability/) is the toy-model version.
- **[Noise Injection Reveals Hidden Capabilities of Sandbagging Language Models](https://arxiv.org/abs/2412.01784)**. Cameron Tice, Philipp Alexander Kreer, Prithviraj Singh Shahani, Nathan Helm-Burger and others (2025). Adds noise to a model's weights and watches whether its score goes up, which a sandbagger's sometimes does. I poked at when that works in [Noise Injection Only Catches Fragile Sandbaggers](/writing/noise-injection-fragile-sandbaggers/).
