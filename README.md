---
title: Content Moderation Safety Benchmark
emoji: shield
colorFrom: red
colorTo: gray
sdk: docker
pinned: false
app_port: 7860
---

# Content Moderation Safety Benchmark (OpenEnv)

[![OpenEnv](https://img.shields.io/badge/Framework-OpenEnv-blue.svg)](https://github.com/meta-llama/openenv)
[![Safety](https://img.shields.io/badge/Domain-AI_Safety-red.svg)](https://hf.co/spaces/anaswarak/content-mod-env)

An OpenEnv environment for evaluating LLM-based content moderation agents. This benchmark focuses on contextual nuance, adversarial phrasing, and multi-turn harassment rather than simple single-label moderation.

---

## System Architecture

```mermaid
graph TD
    A[Agent / LLM] -->|Action: decision + confidence| B[OpenEnv Server]
    B -->|Observation: post + context| A
    B -->|Grader| C{Reward Engine}
    C -->|Calibrated Reward| B
    subgraph Tasks
    D[Direct Toxicity]
    E[Social Bias]
    F[Adversarial]
    G[Multi-Turn Context]
    end
    B -.-> D
    B -.-> E
    B -.-> F
    B -.-> G
```

---

## Key Features

- Confidence-aware scoring for moderation decisions
- Context-sensitive tasks including multi-turn harassment
- OpenEnv-compatible containerized evaluation setup

---

## Evaluation Tiers

| Tier | Name | Difficulty | Description |
| :--- | :--- | :--- | :--- |
| **1** | Direct Toxicity | Standard | Unambiguous hate speech and direct threats. |
| **2** | Social Bias | Medium | Subtle identity-based bias and micro-aggressions. |
| **3** | Adversarial | Advanced | Borderline cases, metaphors, and linguistic traps. |
| **4** | Contextual Harassment | Elite | Harassment spanning multiple conversational turns. |

---

## Setup and Running

### Local Development

```powershell
uv sync
uv run server
$env:HF_TOKEN="your_token"
python inference.py
```

### Docker Execution

```bash
docker build -t content-mod-env .
docker run -p 7860:7860 content-mod-env
```

---

## Reward Logic

Successful moderation is scored with:

```text
Score = Accuracy * (0.7 + 0.3 * Confidence)
```

This rewards calibrated confidence instead of lucky guesses.
