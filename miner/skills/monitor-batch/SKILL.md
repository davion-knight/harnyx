---
name: monitor-batch
description: Monitor waiting, running, and completed miner-task batches without overreading public visibility. Use after submission while tracking cutoff eligibility, running progress, or completion.
---

# Monitor Batch

## Goal

Track whether an accepted artifact is eligible for a batch, whether a batch is
running, and when completed result rows become visible.

## Inputs

- `artifact_id`
- `content_hash`
- submit time
- MCP tools: `get_latest_submissions`, `list_miner_task_batches`, `get_miner_task_batch`

## Steps

1. Call `get_latest_submissions` and confirm the artifact metadata is accepted.
2. Call `list_miner_task_batches` to find candidate batches.
3. Compare `submitted_at` with batch `cutoff_at`.
4. For a running batch, call `get_miner_task_batch(batch_id)` and inspect:
   - batch state
   - delivery state
   - delivery progress
   - validator progress and last error fields when present
5. Wait for completion before looking for artifact rows, result rows, miner
   responses, reference answers, or script content.

## Stop Conditions

- Stop if accepted submission metadata is missing.
- Do not classify a running batch as missing your artifact only because public
  artifact rows are hidden before completion.

## Output

- accepted submission status
- cutoff eligibility assessment
- current batch state
- next monitoring action
