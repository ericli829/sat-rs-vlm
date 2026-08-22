# Qwen3-VL R1 source architecture audit

The architecture is verified from the formal C2 runtime preflight: Qwen3-VL 4B has 24 ViT
blocks, visual width 1024, LLM width 2560, spatial merge 2, and taps 5/11/17/23. DeepStack
features from 5/11/17 are first consumed by LLM layers 1/2/3; tap 23 feeds the layer-0 visual
tokens. The formal preflight passed both frozen R1 base-route parity and C2 step-0 parity.

The local base model and exact processor are present at
`E:/迅雷下载/Qwen3-VL 4B/Qwen3-VL-4B-Instruct`. The processor is Qwen3VLProcessor from
Transformers 5.13.0 with right padding.

Local real-model integration remains blocked because the exact formal R1 adapter and its visual
sidecar were not copied from AutoDL. The required sidecar SHA256 is
`67c2b33d255492080166efc767d1fceb46e007184b162f481f274d5327b989ae`. A different local
adapter or a mock is not an acceptable substitute.

The count-aware path uses an instance-local forward hook on decoder layer 3 and anchors at the
token immediately before the first `labels != -100` token. All 15,428 counting rows retain LM CE;
only strict exact-cardinality rows receive the auxiliary objective. Gradient checkpointing is
accepted only in non-reentrant mode for this path.
