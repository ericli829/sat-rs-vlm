"""Replay captured change questions through the already loaded local 2B service.

No model is loaded here. The caption and region passes use short_text requests,
so their generated answers are saved without the enum likelihood scoring step.
Official gold labels are unavailable; the HTML report supports human review.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


REGIONS = (
    "top_left", "top", "top_right", "left", "center", "right",
    "bottom_left", "bottom", "bottom_right",
)
CAPTION_PROMPT = (
    "The first image is before the change and the second image is after the change. "
    "Describe all visible changes concisely. If there is no change, state that clearly. "
    "Include the locations of objects that appeared, disappeared, or were substantially "
    "altered, using image-relative directions. Ignore differences caused only by lighting, "
    "shadows, seasons, or minor image alignment. Give a brief description of visible "
    "evidence in at most 50 words. Do not select a region label yet."
)
REGION_RULES = (
    "For this experiment, interpret the nine labels as a 3 by 3 grid of equal cells "
    "over the whole image. Top row: top_left, top, top_right. Middle row: left, center, "
    "right. Bottom row: bottom_left, bottom, bottom_right. Left and right refer to the "
    "displayed image, not compass directions. Locate the center of the main actual "
    "structural change; when multiple clusters change, use the cluster with the largest "
    "changed footprint. Check the images directly: the earlier description may be wrong. "
    "Ignore lighting, shadows, seasonal vegetation, and small alignment differences. "
    "Return only one exact region label from the nine listed above, with no punctuation "
    "or explanation."
)
GRID_NOTE = (
    "Experimental assumption: equal 3x3 cells; select the center of the dominant "
    "structural change cluster. The official interface does not define the boundaries "
    "or the primary-change tie-break rule."
)


def now():
    return datetime.now(timezone.utc).isoformat()


def write_text_atomic(path: Path, content: str):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    # Windows readers/scanners can briefly deny replacing an open destination.
    # Retrying the completed rename preserves the previous valid snapshot.
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (2 ** attempt))


def write_json(path: Path, value):
    write_text_atomic(path, json.dumps(value, ensure_ascii=False, indent=2))


def load_cases(capture: Path, source_run: str, score_path: Path, package: Path):
    scores = {
        row["id"]: row["score"]
        for row in json.loads(score_path.read_text(encoding="utf-8-sig"))["details"]
    }
    selected = {}
    for line in capture.read_text(encoding="utf-8-sig").splitlines():
        row = json.loads(line)["request"]
        contract = row["response_constraint"]
        if not row["request_id"].startswith(source_run + ":"):
            continue
        if contract.get("question_form") != "change_region":
            continue
        if (
            contract["type"] != "enum" or set(contract["values"]) != set(REGIONS)
            or len(row["images"]) != 2
        ):
            raise ValueError("Unexpected change-region contract")
        item_id = row["item_id"]
        if item_id in selected and selected[item_id] != row:
            raise ValueError("Conflicting duplicate request: " + item_id)
        selected[item_id] = row
    if not selected:
        raise ValueError("No captured change-region questions for the selected run")
    cases = []
    for item_id, request in sorted(selected.items()):
        images = []
        for descriptor in request["images"]:
            asset = descriptor["asset_id"]
            if not re.fullmatch(r"[0-9a-f]{32}", asset):
                raise ValueError("Invalid asset identifier")
            path = package / "assets" / asset
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != descriptor["sha256"]:
                raise ValueError("Image SHA256 mismatch: " + asset)
            mime = descriptor["mime_type"]
            if mime not in {"image/png", "image/jpeg", "image/webp"}:
                raise ValueError("Unexpected image MIME type")
            images.append({
                "path": str(path.resolve()), "asset_id": asset,
                "sha256": descriptor["sha256"],
                "data_url": "data:" + mime + ";base64," + base64.b64encode(content).decode("ascii"),
            })
        if item_id not in scores:
            raise ValueError("Scoring report does not include " + item_id)
        cases.append({
            "item_id": item_id, "original_request": request, "images": images,
            "website_score": scores[item_id],
        })
    return cases


def generated_request(original, experiment_id, phase, question):
    request = copy.deepcopy(original)
    request["request_id"] = experiment_id + ":" + original["item_id"] + ":" + phase
    request["question"] = question
    if phase != "baseline":
        request["answer_type"] = "short_text"
        request["response_constraint"] = {"type": "short_text", "min_length": 1}
        request.pop("choices", None)
    return request


def valid_region(answer, original):
    if not isinstance(answer, str):
        return None
    answer = answer.strip()
    for region in original["response_constraint"]["values"]:
        if answer == region or (
            original["response_constraint"].get("case_insensitive", False)
            and answer.casefold() == region.casefold()
        ):
            return region
    # Do not extract a region from prose, add a fallback, or guess a label.
    return None


def post(endpoint, secret, request):
    started = time.perf_counter()
    http_request = Request(
        endpoint + "/v1/predict",
        data=json.dumps(request, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": "Bearer " + secret, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(http_request, timeout=135) as response:
            status = response.status
            body = response.read()
    except HTTPError as exc:
        status = exc.code
        body = exc.read()
    payload = json.loads(body)
    result = {"status": status, "response": payload,
              "latency_seconds": round(time.perf_counter() - started, 3)}
    if status == 200:
        if (
            payload.get("request_id") != request["request_id"]
            or payload.get("item_id") != request["item_id"]
            or payload.get("protocol_version") != "2.0"
            or not isinstance(payload.get("answer"), str)
        ):
            raise ValueError("Response identity or answer shape is invalid")
        result["answer"] = payload["answer"]
    return result


def render_report(output: Path, experiment_id, cases, results, stage):
    completed = {row["item_id"]: row for row in results}
    rows = [{**case, "result": completed.get(case["item_id"])} for case in cases]
    data = {
        "experiment_id": experiment_id, "updated_at": now(), "stage": stage,
        "regions": REGIONS, "cases": rows, "grid_note": GRID_NOTE,
        "caption_prompt": CAPTION_PROMPT, "region_rules": REGION_RULES,
    }
    encoded = json.dumps(data, ensure_ascii=True).replace("<", "\\u003c")
    html = REPORT_HTML.replace("__REPORT_DATA__", encoded)
    write_text_atomic(output / "report.html", html)


REPORT_HTML = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><title>变化区域：2B 提示词对照</title>
<style>
body{font:16px/1.55 system-ui,sans-serif;background:#eef2f6;color:#1e293b;margin:0}
main{max-width:1120px;margin:auto;padding:24px}h1{font-size:25px}h2{font-size:19px}
.card,header{background:white;border-radius:12px;padding:20px;margin:16px 0;border:1px solid #dce3ec}
.pictures{display:flex;gap:18px;flex-wrap:wrap}.pictures figure{margin:0;flex:1;min-width:240px}
.photo{position:relative;max-width:410px}.photo img{display:block;width:100%;image-rendering:auto}
.grid{position:absolute;inset:0;display:none;pointer-events:none;background:linear-gradient(to right,transparent calc(33.333% - .5px),#fff 33.333%,transparent calc(33.333% + .5px),transparent calc(66.666% - .5px),#fff 66.666%,transparent calc(66.666% + .5px)),linear-gradient(to bottom,transparent calc(33.333% - .5px),#fff 33.333%,transparent calc(33.333% + .5px),transparent calc(66.666% - .5px),#fff 66.666%,transparent calc(66.666% + .5px))}
body.show-grid .grid{display:block}pre{white-space:pre-wrap;background:#f3f6fa;padding:12px;border-radius:6px;font:14px/1.5 ui-monospace,monospace}
.answers{display:grid;grid-template-columns:1fr 1fr;gap:16px}.muted{color:#64748b;font-size:14px}
button,select,input{font:inherit;padding:6px 10px;margin:4px;border:1px solid #cbd5e1;border-radius:6px}
input{max-width:420px;width:80%}.badge{font-weight:bold;color:#2456a6}details{margin:10px 0}
@media(max-width:650px){.answers{grid-template-columns:1fr}}
</style><main><header><h1>变化区域：2B 提示词对照</h1>
<p>原问题：<b>Which region of the image pair contains the primary change?</b><br>这对图像的主要变化位于哪个区域？</p>
<p id="status"></p><p id="website"></p>
<p class="muted">原网站仅返回判分，未记录当时的具体标签。原路径一栏是本次重新调用的答案。新方法先描述变化，再带着描述和区域规则自由生成标签；没有导入标准答案。人工准确率仅代表你的标注。</p>
<p class="muted" id="grid-note"></p>
<label><input type="checkbox" id="grid-toggle" style="width:auto">显示九宫格辅助线（原图未改动）</label>
<button id="export">导出人工标注 JSON</button><p id="stats"></p>
<details><summary>查看实验提示词</summary><pre id="prompts"></pre></details>
</header><div id="items"></div></main>
<script id="report-data" type="application/json">__REPORT_DATA__</script>
<script>
const data=JSON.parse(document.querySelector('#report-data').textContent);
const storageKey='change-region-review:'+data.experiment_id;
let review={};try{review=JSON.parse(localStorage.getItem(storageKey)||'{}')}catch(e){}
const el=(tag,text,cls)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;};
document.querySelector('#status').textContent=`进度：${data.cases.filter(c=>c.result).length}/${data.cases.length}；状态 ${data.stage}；更新 ${data.updated_at}。运行期间重新打开/刷新本文件获取新结果。`;
const websiteCorrect=data.cases.reduce((s,c)=>s+c.website_score,0);
document.querySelector('#website').textContent=`原网站变化题：${websiteCorrect}/${data.cases.length} = ${(websiteCorrect/data.cases.length*100).toFixed(1)}%。`;
document.querySelector('#grid-note').textContent='实验假设：把整图均分成九宫格，用主要结构变化区域的中心选标签；多处变化取变化面积最大的区域。官方文档未规定这些边界和规则。';
document.querySelector('#prompts').textContent='Step 1 — caption\n'+data.caption_prompt+'\n\nStep 2 — region\n[Original question]\n[Earlier model caption]\n'+data.region_rules+'\n\n服务还会追加 BEFORE/AFTER 说明及最简直接回答要求。生成上限沿用服务的 96 tokens。';
function save(){try{localStorage.setItem(storageKey,JSON.stringify(review))}catch(e){}updateStats()}
function updateStats(){let reviewed=0,excluded=0,ready=0,b=0,g=0,valid=0;for(const c of data.cases){const r=review[c.item_id]||{};if(r.region==='unclear'){excluded++;continue}if(!data.regions.includes(r.region))continue;reviewed++;if(!c.result)continue;ready++;b+=Number(c.result.baseline_region===r.region);g+=Number(c.result.guided_region===r.region);valid+=Number(Boolean(c.result.guided_region))}const pct=x=>ready?(x/ready*100).toFixed(1)+'%':'待标注';document.querySelector('#stats').textContent=`已人工标注 ${reviewed}，歧义/无明显变化 ${excluded}；已有结果的标注题 ${ready}。原路径 ${b}/${ready}（${pct(b)}）；新方法 ${g}/${ready}（${pct(g)}）；新方法标签有效 ${valid}/${ready}。无效输出按错误计。`;}
function outputText(pass){return !pass?'尚未返回':pass.answer!==undefined?pass.answer:JSON.stringify(pass.response||pass)}
for(const c of data.cases){const card=el('section',undefined,'card');card.append(el('h2',c.item_id));const pics=el('div',undefined,'pictures');c.images.forEach((im,i)=>{const fig=el('figure');fig.append(el('figcaption',i===0?'变化前 BEFORE':'变化后 AFTER'));const photo=el('div',undefined,'photo');const img=el('img');img.src=im.data_url;img.alt=c.item_id+(i===0?' before':' after');photo.append(img,el('div',undefined,'grid'));fig.append(photo);pics.append(fig)});card.append(pics);
const answers=el('div',undefined,'answers');const baseline=el('div');baseline.append(el('h3','原路径（本次重放）'),el('pre',c.result?outputText(c.result.baseline):'尚未返回'));const guided=el('div');guided.append(el('h3','描述后自由生成'),el('pre',c.result?outputText(c.result.guided):'尚未返回'));answers.append(baseline,guided);card.append(answers);
card.append(el('p','模型描述的可见变化：','badge'),el('pre',c.result?outputText(c.result.caption):'尚未返回'));
const r=review[c.item_id]||{};const label=el('label','你判断的主要变化区域：');const select=el('select');[['','尚未判断'],...data.regions.map(x=>[x,x]),['unclear','歧义 / 无明显变化']].forEach(([v,t])=>{const o=el('option',t);o.value=v;select.append(o)});select.value=r.region||'';select.addEventListener('change',()=>{review[c.item_id]={...(review[c.item_id]||{}),region:select.value};save()});label.append(select);card.append(label);const notes=el('input');notes.placeholder='标注备注（可选）';notes.value=r.notes||'';notes.addEventListener('input',()=>{review[c.item_id]={...(review[c.item_id]||{}),notes:notes.value};save()});card.append(notes);
const details=el('details');details.append(el('summary','查看原请求、网站判分及耗时'));details.append(el('pre',JSON.stringify({original_request:c.original_request,website_score:c.website_score,result:c.result},null,2)));card.append(details);document.querySelector('#items').append(card)}
document.querySelector('#grid-toggle').addEventListener('change',e=>document.body.classList.toggle('show-grid',e.target.checked));
document.querySelector('#export').addEventListener('click',()=>{const exported={experiment_id:data.experiment_id,exported_at:new Date().toISOString(),grid_note:data.grid_note,review,cases:data.cases.map(c=>({item_id:c.item_id,baseline_region:c.result?.baseline_region||null,guided_region:c.result?.guided_region||null}))};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(exported,null,2)],{type:'application/json'}));a.download=data.experiment_id+'-human-review.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)});updateStats();
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.jsonl").exists():
        raise ValueError("Use a fresh output directory; do not overwrite a previous experiment")
    state = json.loads(args.state.read_text(encoding="utf-8-sig"))
    endpoint = state["endpoint"].rstrip("/")
    origin = urlsplit(endpoint)
    if origin.scheme != "http" or origin.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("This experiment only calls the authenticated local service")
    cases = load_cases(args.capture, args.source_run, args.score, args.package.resolve())
    prepared = output / "experiment.json"
    experiment_id = (
        json.loads(prepared.read_text(encoding="utf-8"))["experiment_id"]
        if prepared.exists() else "change-region-" + uuid.uuid4().hex
    )
    write_json(prepared, {
        "experiment_id": experiment_id, "source_run": args.source_run,
        "source_capture": str(args.capture.resolve()), "source_score": str(args.score.resolve()),
        "endpoint": endpoint, "created_at": now(), "grid_note": GRID_NOTE,
        "caption_prompt": CAPTION_PROMPT, "region_rules": REGION_RULES,
        "model_config": json.loads(Path(state["config_file"]).read_text(encoding="utf-8-sig"))["model"],
        "cases": [{k: v for k, v in c.items() if k != "images"} for c in cases],
    })
    results = []

    def stage(name, **extra):
        write_json(output / "status.json", {
            "experiment_id": experiment_id, "stage": name, "updated_at": now(),
            "total": len(cases), "completed": len(results), **extra,
        })

    render_report(output, experiment_id, cases, results, "prepared")
    stage("prepared")
    if args.prepare_only:
        print(json.dumps({"prepared": len(cases), "report": str(output / "report.html")}), flush=True)
        return
    secret = Path(state["key_file"]).read_text(encoding="utf-8-sig").strip()
    if not secret:
        raise ValueError("Empty local API key")
    try:
        with (output / "responses.jsonl").open("a", encoding="utf-8", buffering=1) as raw, (
            output / "results.jsonl"
        ).open("a", encoding="utf-8", buffering=1) as records:
            for case in cases:
                original = case["original_request"]
                item_id = case["item_id"]

                def call(phase, question):
                    stage(phase, item_id=item_id)
                    request = generated_request(original, experiment_id, phase, question)
                    result = post(endpoint, secret, request)
                    raw.write(json.dumps({"received_at": now(), "phase": phase,
                        "request": request, **result}, ensure_ascii=False) + "\n")
                    raw.flush()
                    if result["status"] in {401, 503, 504}:
                        raise RuntimeError("Local service returned HTTP " + str(result["status"]))
                    return result

                baseline = call("baseline", original["question"])
                caption = call("caption", CAPTION_PROMPT)
                guided = None
                if caption.get("answer"):
                    guided = call("guided", original["question"] + "\n\nEarlier model description "
                        "(unverified):\n" + caption["answer"] + "\n\n" + REGION_RULES)
                result = {
                    "item_id": item_id, "finished_at": now(), "baseline": baseline,
                    "caption": caption, "guided": guided,
                    "baseline_region": valid_region(baseline.get("answer"), original),
                    "guided_region": valid_region((guided or {}).get("answer"), original),
                    "website_score": case["website_score"],
                }
                records.write(json.dumps(result, ensure_ascii=False) + "\n")
                records.flush()
                results.append(result)
                render_report(output, experiment_id, cases, results, "running")
                stage("running", last_item_id=item_id)
                print(json.dumps({"completed": len(results), "total": len(cases),
                    "item_id": item_id, "baseline": result["baseline_region"],
                    "guided": result["guided_region"]}), flush=True)
        valid = sum(bool(row["guided_region"]) for row in results)
        write_json(output / "summary.json", {
            "count": len(results), "guided_valid": valid,
            "website_correct": sum(c["website_score"] for c in cases),
            "new_official_accuracy": None,
            "accuracy_note": "No official gold labels. Use report.html to annotate and export human review.",
        })
        render_report(output, experiment_id, cases, results, "complete")
        stage("complete", guided_valid=valid, report=str(output / "report.html"))
    except BaseException as exc:
        render_report(output, experiment_id, cases, results, "failed")
        stage("failed", error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == "__main__":
    main()
