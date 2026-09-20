# 09: Demo-oriented reappearance suite

Six three-shot cases follow the paper's prompt construction: establish one visually
distinctive target, move the camera to a different region so the target leaves the visible
scene, then return while referring only to the same target. The third prompt deliberately
does not repeat the key appearance attributes.

Each case is generated with the same seed under three matched schemes:

- `baseline`: LongLive-2.0 local/sink context without LayerRecall;
- `layerrecall`: released LayerRecall checkpoint, cosine retrieval, profiled ten layers;
- `agent_event`: the agent identifies Shot 1 as the valid historical event, while cosine
  retrieval still selects two complementary chunks independently inside each profiled layer.

LayerRecall and agent-event use the same 160-frame physical cache and the same two-chunk
visible memory budget. The larger cache isolates retrieval quality for this short demo suite.
All videos use the L20 screening resolution (approximately 384 x 640 decoded).
