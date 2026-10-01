# label-consistency

Labels the same agent messages twice with differently worded prompts and reports how often a small model agrees with itself, the margin to read any small-model breakdown with.

Counting is arithmetic; deciding what messages *are* is inference, and every call
is metered into `spend.jsonl`. Nothing is published unless `--publish` is passed.

```
python agent.py                 # measure, classify, print, publish nothing
python agent.py --publish       # ... and post the finding and the note
```

Inference auto-detects Ollama on localhost; set `AGENT_MODEL` to pick a model.
