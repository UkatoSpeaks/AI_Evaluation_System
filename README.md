# AI Evaluation Agent

A refund-support agent built with LangChain (running on Groq), evaluated with
[DeepEval](https://github.com/confident-ai/deepeval) using Gemini as the judge model.

## Layout

```
agent/
  agent.py        create_agent setup + interactive CLI (run_conversation)
  tools.py        purchase lookup, date, refund calculation (enforces policy), ticket
  data.py         mock customers and purchases (today is fixed to 2026-10-28)
  prompts.py      system prompt with the refund policy
  objects.py      tool input types (PurchaseItem)
  models.py       agent LLM (Groq)
  list_models.py  print the models your Groq key can use
evaluations/
  eval_all.py     runs the agent over the dataset and scores it (all metrics by default)
  metrics.py      every metric's configuration (thresholds, task description)
  common.py       shared helpers: env setup, run_agent with retries, tool-call extraction
  dataset.py      the goldens
  model.py        judge LLM (Gemini)
  <metric>.py     shortcuts that run a single metric
```

## Setup

```bash
uv sync
cp .env.example .env   # then fill in GROQ_API_KEY and GOOGLE_API_KEY
```

## Run

```bash
# All 7 metrics
uv run python evaluations/eval_all.py

# A subset
uv run python evaluations/eval_all.py --metrics tool_correctness answer_correctness

# One metric
uv run python evaluations/tool_correctness.py

# Chat with the agent
uv run python agent/agent.py
```

Metrics: `task_completion`, `step_efficiency`, `plan_quality`, `plan_adherence`
(trace-based), and `argument_correctness`, `tool_correctness`, `answer_correctness`
(single-turn).

A full run takes a while: there is a 20s pause between goldens for Groq's
rate limit, and the Gemini judge retries 429s with backoff.
