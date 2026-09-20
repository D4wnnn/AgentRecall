# 08: Seed-1 replication

This is the stopping-gate replication for experiments 06--07. At seed 1 and with the
entire 160-frame sequence physically resident, compare cosine retrieval, the planned early
event chunks `[2, 1]`, and the planned late event chunks `[7, 6]`. All visible-memory and
layer budgets are identical.

Stop and promote event/key-state planning if early planning again improves final return-shot
attribute stability without worse temporal diagnostics. Otherwise report the seed-0 result
as unstable and do not escalate to an LLM/VLM agent yet.
