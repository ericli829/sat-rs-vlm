# VLM-FO1 official source audit

- repository: `https://github.com/om-ai-lab/VLM-FO1`
- commit: `348c1e8163a8fca5ed621cdfab0c94e3432336bd`
- model: `omlab/VLM-FO1-3B-v01` (Qwen2.5-VL-3B)
- architecture: `OmChatQwen25VLForCausalLM`
- local weight files present: `False`

## Public proposal path

- generator: `UPNWrapper from detect_tools/upn`
- score threshold: `0.3`
- NMS threshold: `0.8`
- top-k: `100`
- boxes: `pixel xyxy in original image coordinates`
- OPN status: OPN referenced in the paper is not released; official public path uses UPN.

## Official FO1 call

- template: `How many {target} are there in this image? Count each instance of the target object. Locate them with object indexes and then answer the question with the number of objects.`
- output: `<ground>label</ground><objects><regionN>...</objects>`
- generation: `max_tokens=4096`, `top_p=0.05`, `temperature=0.0`, `do_sample=False`
- tokenizer: slow tokenizer; region tokens `<region0>` through `<region99>`

## Vision sidecars

- primary tower: `resources/Qwen2.5-VL-3B-Instruct-Vision_Tower`
- auxiliary tower: `resources/davit-large.pth`
- auxiliary size/aspect: `1024` / `dynamic`

## Isolation

The official requirements are installed only in `vlm-fo1`; the rs-vlm interpreter communicates through JSONL and never imports the official package.

## Recorded differences

- README and scripts expose both Hugging Face and local model paths; the evaluator requires an explicit local VLM_FO1_MODEL.
- The paper mentions OPN, while the public repository documents UPN as the available object proposal path.
- The official script calls filter(min_score=0.3), whose implementation default nms_value=0.8 is recorded explicitly here.
