# Paragraph-level Writing Experiments

> **Archived on November 2, 2026.** This repository preserves exploratory Korean
> writing-diagnosis and revision experiments. Development has ended.

The project compares two ways to revise an essay:

- **Whole essay (`whole_once`):** diagnose, summarize revision needs, and revise the
  complete essay in one pass.
- **Incremental (`incremental`):** revise one paragraph at a time, using previously
  revised text and a memory of earlier revisions to guide the next paragraph.

Kanana scores and linguistic features provide diagnostic evidence. A summarizer
prepares a revision brief, a reviser applies it, and the incremental condition
updates a shared profile and revision memory. The experiments examine paragraph
connections, style consistency, meaning preservation, and the effects of memory.

## Current projects

| Repository | Focus |
|---|---|
| [FEAK-Agent](https://github.com/grrlkk/FEAK-Agent) | Orchestrator, Korean, and Revision agents; shared essay state and corruption-data generation. |
| [LIFT](https://github.com/grrlkk/LIFT) | Learning-oriented iterative writing feedback with a Coordinator, Feedback agent, and shared learner profile. |

## Archived implementation

| Path | Contents |
|---|---|
| `src/para_agent/run/` | Whole-essay and incremental experiment runners |
| `src/para_agent/agents/` | Summarization and revision roles |
| `src/para_agent/diagnose/` | Scorer and linguistic-feature adapters |
| `src/para_agent/memory/` | Profile and revision-memory handling |
| `src/para_agent/eval/` | Scoring and judge-based evaluation |
| `prompts/`, `configs/`, `scripts/` | Experiment prompts, settings, and entry points |

## Historical documentation

The [last active README](https://github.com/grrlkk/para_agent/blob/1e323b9f324552925345d46360f9cdf2000e2a8c/README.md)
records the original setup, pilot experiments, evaluation results, and unresolved
limitations. Pilot score changes should be read together with the reported meaning-
preservation and evaluation issues. Source history and experiment documentation are
retained for reference.
