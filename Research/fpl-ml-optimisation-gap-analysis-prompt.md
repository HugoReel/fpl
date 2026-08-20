# Prompt for Claude (Fable 5) – Gap Analysis of FPL ML + Optimisation Blueprint

You are an expert in:

- machine learning for sports/football analytics,
- mathematical optimisation (integer programming),
- Fantasy Premier League (FPL) rules and strategy,
- production ML systems and backtesting frameworks.

I will give you a research and implementation blueprint for an end‑to‑end FPL system that:

- predicts FPL player points,
- selects optimal squads and starting XIs,
- plans transfers over a season, and
- decides when to use chips/boosters (wildcard, free hit, bench boost, triple captain).

Your task is to **critically analyse this blueprint and identify key gaps, weaknesses, and missing pieces** that would matter for building a serious, production‑grade system.

---

## Context

The blueprint is structured into these main sections:

1. FPL rules and optimisation problem formalisation  
2. Data requirements and sources  
3. Target definition and feature engineering  
4. Modelling approaches and training setup  
5. Squad, transfer, and chip optimisation  
6. Backtesting on historical seasons  
7. Compute requirements and tech stack  
8. Review of existing FPL ML and optimisation projects  
9. Additional dimensions to research  
10. Phased implementation roadmap  

Assume the intended user is a technically literate solo developer (comfortable with Python, basic ML and optimisation) who wants to implement this over a few months.

---

## Your Task

Read the blueprint carefully and produce a **structured gap analysis** that:

1. **Summarises the blueprint’s strengths** (what it gets right, what is well‑covered).
2. **Identifies key gaps and weaknesses**, grouped by theme, for example:
   - Data and feature gaps  
   - Modelling and validation gaps  
   - Optimisation formulation gaps  
   - Backtesting and evaluation gaps  
   - Compute, engineering and MLOps gaps  
   - Strategic / FPL‑specific gaps (e.g. chips, double/blank GWs, mini‑league vs OR trade‑offs)
3. For each gap, explain:
   - **Why it matters** (impact on performance, robustness, or usability),
   - **What a good solution might look like** (concrete ideas, approaches, or references),
   - **Priority**: “critical for v1”, “important but can wait”, or “nice‑to‑have”.
4. Highlight any **over‑optimistic assumptions** or **under‑specified areas** (e.g. “assumes perfect injury info”, “glosses over price dynamics”, “no clear way to handle double GW uncertainty”).
5. Suggest **2–3 high‑leverage research questions or experiments** that, if answered, would most improve the system’s design before implementation.
6. Optionally, propose **adjustments to the phased roadmap** (v0.1, v0.2, v0.3, v1.0) to better address the most critical gaps.

Be specific, technical, and pragmatic. Prefer concrete, actionable recommendations over vague advice.

---

## Input: Blueprint Text

Below is the full blueprint text. Use it as your sole context for this analysis.

```text
[PASTE THE FULL BLUEPRINT REPORT HERE]
```

---

## Output Format

Structure your answer as:

1. **Brief overview** (3–6 sentences) of your overall assessment.  
2. **Strengths of the blueprint** (bullet list).  
3. **Key gaps and weaknesses**, grouped by theme, each with:
   - Gap description  
   - Why it matters  
   - Suggested approach / remedy  
   - Priority (critical / important / nice‑to‑have)  
4. **Over‑optimistic or under‑specified assumptions** (bullet list).  
5. **High‑leverage research questions / experiments** (3–5 items).  
6. **Suggested roadmap adjustments** (optional, bullet list).  

Keep the tone constructive but rigorous.