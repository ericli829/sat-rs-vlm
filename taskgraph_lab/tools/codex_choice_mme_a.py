# ruff: noqa: E501
# Keep finite reviewed English descriptions and annotation rows intact for audit.
"""Reviewed MME shard A planning labels; no source answer access or model call.

Each numbered specification below was assigned after reading the corresponding
real question. The renderer only applies these reviewed specifications; it is
not a general question-to-gold keyword labeller.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from taskgraph_lab.taskgraph.canonicalize import canonicalize_target
from taskgraph_lab.taskgraph.dsl import compile_taskgraph_to_dsl, parse_taskgraph_dsl
from taskgraph_lab.taskgraph.validator import validate_candidate

ROOT = Path(__file__).resolve().parents[1] / "data" / "choice_expansion_codex_v1"
REVIEWED_SOURCE_SHA256 = "163b3e08838ec8bf34c1a65135566fa767a19ec53b5cfbb3f8c2c222adee60a7"

# index|category|intrinsic key=value;...|absolute image region|selection|part
# S/M = semantic SINGLE/MULTI with the original complete target description.
# E/R = deterministic extreme/area rank; AROUND/NEXT_TO = contextual subregion.
REVIEWED = r"""
0|roof||TOP_RIGHT|S
1|boat parking area|shape=square||S
2|roof|||LARGEST_ROOF
3|building|shape=L-shaped;has_part=triangular roof|TOP_RIGHT
4|car|state=parked||M
5|car||TOP_LEFT|S
6|hoisting equipment||TOP_LEFT|M
7|Ping-Pong table||TOP_RIGHT|S
8|installation|color=black;shape=circular||M
9|building|has_part=green arched roof||M
10|boat|color=red||M
11|parasol||TOP_LEFT|M
12|concrete mixing tower||CENTER|M
13|building|||S|roof
14|flag||TOP_RIGHT|S
15|building|has_part=two triangular ridges|TOP_RIGHT||roof
16|house|has_part=dome|TOP_LEFT
17|building complex|shape=8-shaped|BOTTOM_CENTER
18|car|||TOP_ROAD
19|building|||S|roof
20|building|shape=square;has_part=striped roof|BOTTOM_RIGHT
21|cargo ship||BOTTOM_RIGHT|LARGER_ADJACENT
22|green space|||R_DESC_AROUND
23|cylinder|color=red|BOTTOM_LEFT|NEXT_TO
24|container||BOTTOM_LEFT|M
25|wind turbine||TOP_LEFT|S
26|disc|||S
27|island|size=small||S
28|building|shape=octagonal||S
29|object|shape=T-shaped|LEFT|S
30|building||BOTTOM_LEFT|M|roof
31|truck||LEFT|S
32|flower||LEFT|M
33|grid object|shape=rectangular|BOTTOM_RIGHT|AROUND
34|ship||BOTTOM|M
35|coastal structure|shape=U-shaped|TOP_RIGHT|S
36|ship|size=large;state=docked|LEFT|S
37|flag|has_part=white pole||M
38|building|shape=cylindrical;has_part=white top||AROUND
39|tennis court||TOP_RIGHT|S
40|board|pattern=striped||S
41|ship|size=large|RIGHT|S
42|house|has_part=yellow roof||M
43|house|||E_LEFTMOST
44|building|has_part=white round roof and two rectangular decorations|TOP_RIGHT
45|car|state=parked|TOP_LEFT|S
46|boat|color=red||M
47|vehicle||TOP_LEFT|S
48|square|color=blue|BOTTOM|NEXT_TO
49|kiddie ride||BOTTOM_LEFT|S|ground
50|truck||BOTTOM_CENTER|LARGEST_ROAD
51|building|||S
52|car||TOP_RIGHT|S
53|flag|||S
54|ship|||S
55|car||TOP_RIGHT|M
56|lawn|color=green;shape=circular||M
57|car|||S
58|structure|color=silver;shape=circular||M
59|square||TOP|S
60|house||BOTTOM_RIGHT|S
61|building|color=white;has_part=circular roof|TOP_RIGHT
62|car||TOP_RIGHT|S
63|parasol|||M
64|area|pattern=two connected red circles
65|car|color=white||M
66|football field|||M
67|crane||BOTTOM_LEFT|S
68|building||TOP_LEFT|M|roof
69|car|color=white|LEFT|M
70|house|has_part=square roof||M
71|traffic roundabout|||S
72|boat||TOP|M
73|truck container||TOP_LEFT|S
74|building|shape=circular|CENTER_LEFT|M
75|house|has_part=white curved roof||S
76|roof|has_part=pointy decoration|TOP_RIGHT
77|car||TOP_RIGHT|S
78|boat||CENTER|M
79|pier||CENTER|S
80|structure|color=white;shape=cylindrical|CENTER|M
81|house|color=purple||S
82|car||BOTTOM_LEFT|S
83|pattern|size=small;shape=rectangular|RIGHT|S|fill
84|car|state=parked|BOTTOM_LEFT|S
85|van||TOP_RIGHT|S
86|parasol|color=yellow|CENTER_LEFT|M
87|building|has_part=gray dome||S
88|tree||BOTTOM_RIGHT|S
89|car||BOTTOM_LEFT|S
90|wall||TOP|M
91|building|shape=U-shaped|TOP_RIGHT|S
92|board||CENTER|S
93|boat|color=green||M
94|object|color=white;shape=cylindrical|BOTTOM_LEFT|NEXT_TO
95|ship||RIGHT|M
96|boat||TOP_RIGHT|S
97|concrete mixing tower||CENTER_RIGHT|M
98|wall|pattern=graffiti|TOP
99|building|color=blue||S
100|car|color=yellow|BOTTOM_RIGHT|NEXT_TO
101|building|shape=rectangular|TOP_RIGHT|S
102|house|||S|roof
103|boat|||S
104|traffic sign||RIGHT|M
105|parking lot|state=empty||S
106|cylinder||BOTTOM|S
107|shelf|color=purple||S
108|house|has_part=green roof|TOP_LEFT
109|building|color=green;shape=cylindrical||M
110|house|has_part=a lot of gray glass on the roof
111|car|||TOP_ROAD
112|ship||RIGHT|M
113|house||TOP_LEFT||roof
114|building||BOTTOM_RIGHT|R_DESC|roof
115|excavator|color=yellow
116|building|has_part=black and white striped roof
117|ship||BOTTOM_RIGHT|S
118|deck|color=orange
119|pattern|color=yellow;shape=circular;pattern=Yale lettering
120|structure|color=dark green;shape=rectangular|CENTER_RIGHT|M
121|building|shape=L-shaped
122|building|color=blue and white
123|football field
124|roof|shape=L-shaped|BOTTOM_LEFT
125|tower|size=tall|TOP|M
126|ground|color=white;pattern=grid
127|helipad|||S
128|ship||TOP_LEFT|R_DESC
129|roof|shape=rectangular|BOTTOM_LEFT
130|car|color=white|TOP_LEFT
131|car|color=coffee;pattern=double white stripes
132|boat|||E_RIGHTMOST
133|roof|shape=square|TOP_RIGHT
134|building|has_part=square yellow roof
135|ship|color=red and blue;size=large
136|sports field
137|cargo ship|color=blue||R_DESC
138|ship|size=large|TOP_CENTER
139|boat||LEFT|M
140|car||BOTTOM_LEFT|S
141|sculpture|color=dark gray
142|building|has_part=red L-shaped roof
143|building|shape=L-shaped|RIGHT||roof
144|facility|color=yellow
145|helipad|shape=circular
146|car|color=white|TOP|M
147|water area|size=small;shape=square
148|object|shape=round|TOP_CENTER
149|playground track
150|building||BOTTOM_LEFT||roof
151|recreation facility||BOTTOM_RIGHT|S
152|boat|color=red;size=small
153|boat||RIGHT|S
154|building|has_part=circular dome|TOP
155|parking area|color=yellow
156|dome|color=pale green
157|car||TOP_LEFT
158|car|color=blue;state=parked alone
159|roundabout|color=green
160|building|||ROW_ROAD|roof
161|building|has_part=round roof|TOP
162|ship|color=red;size=large|TOP_LEFT
163|building|color=white
164|truck|has_part=double box|RIGHT|M
165|car||BOTTOM_LEFT|LEFTMOST_ROAD
166|boat or yacht|color=yellow||M
167|car|color=red|BOTTOM_CENTER|M
168|building|has_part=red and white circular roof
169|building|has_part=blue roof|BOTTOM_RIGHT
170|ship|has_part=red and orange containers
171|logo|pattern=McDonald's
172|ground|pattern=blue and yellow circular patterns
173|parasol||TOP_RIGHT|M
174|building|has_part=orange partition on roof
175|building|color=brown and blue
176|traffic roundabout|||S
177|building|color=white;shape=column-shaped||GROUP_CLUSTER
178|letter||LEFT|M
179|logo|||S
180|grass|color=green;pattern=two oval patterns
181|building|has_part=brick red wall
182|car|state=parked|TOP_RIGHT|M
183|pipeline||RIGHT|S
184|roof|shape=triangular|BOTTOM_RIGHT
185|ship|size=small;state=moving|BOTTOM_LEFT|S
186|boat||TOP_RIGHT
187|cargo ship|||LARGEST_RIVER
188|building complex|shape=U-shaped;has_part=red roof
189|boat or yacht||TOP_RIGHT|M
190|circle||BOTTOM_CENTER
191|playground|pattern=circular ground pattern
192|bridge|||S
193|car|color=black;state=moving|TOP_RIGHT|M
194|structure|pattern=blue yellow and green alternating colors
195|building|has_part=colored billboards on front exterior wall
196|building|has_part=blue roof with white cross pattern
197|sign||TOP_RIGHT|S
198|roof|shape=rectangular|TOP|S
199|building|color=black;shape=L-shaped
200|bus|color=white
201|watchtower|color=red and white
202|cargo||TOP|S
203|house|color=gray;shape=T-shaped
204|ship|color=blue and white;size=large
205|building|size=large;has_part=blue roof
206|board|color=purple;shape=square
207|boat or yacht||BOTTOM|M
208|truck|color=green
209|area|shape=fan-shaped
210|ship|size=large
211|roof|color=gray;has_part=cylindrical structures
212|parking lot
213|boat|size=large|TOP_RIGHT
214|sculpture|color=light green;shape=pentagonal
215|car|color=red|TOP_LEFT|M
216|playground|color=green
217|lawn|color=green;pattern=white hexagonal pattern
218|house|color=light green
219|roof|shape=L-shaped;pattern=white edge
220|building|size=tall|TOP_LEFT
221|ship|||S
222|slide|||S
223|ship||TOP_LEFT
224|road sign||TOP_LEFT|S
225|ground area|color=green;shape=rectangular;pattern=two white dashed lines
226|building|color=white;shape=chimney-shaped
227|playground|has_part=red tracks
228|house||TOP_LEFT
229|tramcar
230|truck|state=moving|BOTTOM_RIGHT|S
231|lawn section||BOTTOM_RIGHT|M
232|ship
233|building|shape=triangular|TOP_LEFT||roof
234|tent||TOP_RIGHT|S
235|ship||TOP_LEFT|M
236|building|color=red|TOP_LEFT||roof
237|car|color=white|TOP_LEFT|M
238|boat||BOTTOM_RIGHT|S
239|ship||TOP_RIGHT|R_ASC
240|ship||BOTTOM_RIGHT
241|tennis court||TOP_LEFT
242|roof|color=silver;shape=spherical|RIGHT|M
243|building|color=gray;shape=square;pattern=white pattern
244|car||BOTTOM_LEFT|M
245|bus|color=red;has_part=white roof
246|car||TOP_RIGHT
247|building|has_part=white-roofed attic
248|ship||BOTTOM_LEFT
249|ship|state=docked|TOP_RIGHT
250|island|color=green|BOTTOM|M
251|car|state=moving|TOP_RIGHT|M
252|building|color=gray-white;shape=triangular
253|hoisting structure||TOP_CENTER|M
254|ground area|color=multicolored
255|reservoir|shape=circular;has_part=long white radius in the middle
256|parasol||TOP_LEFT
257|object|shape=circular|TOP_CENTER|M
258|roof|shape=cylindrical
259|car|state=driving downwards|BOTTOM_RIGHT|S
260|car|color=red|BOTTOM_LEFT|M
261|pattern|pattern=checkerboard
262|car|state=moving|BOTTOM_LEFT|S
263|building|pattern=wi. wandenden strack lettering
264|intersection|has_part=rainbow-colored zebra crossings
265|bus|size=large;state=moving|TOP_RIGHT|S
266|car|||M
267|pitch|color=gray
268|open space|color=green;shape=rectangular
269|basketball court||TOP_RIGHT
270|tree|has_part=pink flowers
271|boat||BOTTOM
272|flag||TOP_LEFT|M
273|car||BOTTOM_RIGHT|M
274|basketball court|color=blue
275|block|color=blue;shape=rectangular
276|boat|color=green
277|vehicle|color=white;state=driving to upper right|TOP_LEFT|M
278|playground
279|ship|size=large|TOP_RIGHT
280|boat|color=black
281|island
282|river confluence
283|square|pattern=blue red and yellow stripes|CENTER_LEFT
284|boat|color=red;size=large;state=docked|LEFT|M
285|ship|color=yellow-green
286|sports field|pattern=colored patterns
287|building|has_part=blue roof||R_DESC
288|building|color=white||S
289|truck|state=moving|TOP_RIGHT|S
290|road|pattern=white number 30 on ground
291|church or chapel
292|building|has_part=red exterior wall
293|residential area
294|building||CENTER|M
295|ship||BOTTOM_LEFT
296|sun umbrella|color=green red and purple
297|truck|size=large;has_part=white front and red compartment
298|structure|shape=cylindrical|TOP_RIGHT
299|building|color=sand|BOTTOM_LEFT
300|island|color=white|TOP_LEFT
301|building|||S
302|house|color=yellow|||roof
303|sunshade|color=red and white
304|house||TOP_LEFT||roof
305|building|shape=arc-shaped
306|truck|size=large||S
307|cargo ship|color=green
308|boat|color=white|BOTTOM_RIGHT|M
309|car||BOTTOM_RIGHT
310|square|size=small;has_part=statue
311|building|shape=L-shaped|TOP_LEFT||roof
312|pattern|color=green;shape=octagonal|BOTTOM_RIGHT
313|freighter|size=large|LEFT|S
314|car|color=yellow||S
315|grassland|color=green;has_part=pale pink flowering trees
316|parking area|||S|ground
317|football field|||S
318|building|shape=E-shaped;has_part=blue roof
319|roof|color=blue
320|ship|||R_ASC
321|building||TOP_LEFT
322|object|color=gray;shape=cylindrical|BOTTOM_LEFT
323|ship|color=red and green;size=large
324|park|shape=triangular
325|building|color=red||S
326|building|has_part=green roof
327|car|color=white|TOP_RIGHT
328|runway|shape=elliptical|TOP_LEFT
329|house|has_part=bright pink roof
330|traffic roundabout
331|buoy||BOTTOM_RIGHT|S
332|van|color=green
333|building|color=white|BOTTOM_RIGHT|INSIDE
334|car|color=yellow
335|house||TOP_LEFT
336|tennis court||TOP_RIGHT
337|van||BOTTOM|E_LEFTMOST
338|meadow|shape=round
339|circle|size=large;has_part=metal surface||S
340|building|has_part=white roof
341|ship||CENTER|S
342|track and field
343|building|color=gray;shape=cylindrical|BOTTOM_RIGHT|M
344|grassland
345|car||BOTTOM_LEFT|S
346|intersection|||R_DESC
347|basketball court||BOTTOM
348|building|shape=cage-shaped;pattern=Tropicana logo
349|boat|||S
350|engineering tower
351|building|pattern=letter P on wall
352|ship|size=large;has_part=green helipad
353|ship|||DOCK_ORDINAL
354|boat||BOTTOM_RIGHT|M
355|building|shape=cylindrical
356|car|state=turning|BOTTOM_CENTER|S
357|room|size=small;pattern=vanUden lettering|BOTTOM_LEFT
358|buoy|color=green
359|sailboat|color=white|TOP_RIGHT|M
360|car||TOP_LEFT|M
361|airplane|color=blue
362|playground|state=outdoor;has_part=children's play equipment
363|sign||TOP|S
364|building|has_part=blue roof
365|slide|color=red and white
366|ship|state=sailing|LEFT|S
367|roof|color=blue;pattern=two white rectangles
368|bridge|||R_DESC
369|car||BOTTOM_LEFT|M
370|playground
371|lawn|color=green;pattern=yellow circular pattern
372|building||TOP_LEFT
373|sandpit
374|fence||TOP_RIGHT|S
375|building||TOP_RIGHT|S
376|traffic roundabout|color=green
377|boat|color=blue|TOP
378|truck||BOTTOM_LEFT
379|square|pattern=rainbow-colored patterns
380|square|pattern=red and white mosaic grid
381|car||CENTER
382|building|shape=symmetrical U-shaped
383|railway
384|boat|pattern=round symbol on stern
385|building|has_part=black and white striped roof
386|tent|size=small|TOP_LEFT
387|parking lot|||R_DESC
388|car||TOP_RIGHT|S
389|structure|color=silver-white;shape=conical|TOP_LEFT|M
390|roof||BOTTOM_RIGHT
391|excavator||BOTTOM_RIGHT|S
392|house||TOP_RIGHT
393|excavator||TOP_RIGHT|S
394|car|color=yellow
395|building|||M
396|building|||S
397|water area|shape=circular
398|container truck|color=yellow
399|pattern|shape=oval|TOP_LEFT|M
400|traffic roundabout|pattern=red circular ring
401|house|color=red
402|field|color=red;has_part=blue box
403|building|has_part=dark blue roof
404|rectangle||TOP
405|cargo||TOP_RIGHT
406|tent|has_part=red and white roof
407|traffic roundabout|color=green;has_part=flowers
408|building|has_part=orange roof
409|structure|color=light brown;shape=cylindrical|BOTTOM_RIGHT
410|bus|color=white
411|object|state=reflective|TOP_LEFT
412|mixer truck
413|building|shape=circular|BOTTOM_LEFT
414|building complex|has_part=red roof|TOP_LEFT||exterior walls
415|grassland|shape=triangular;pattern=blue circular pattern
416|ship||TOP_LEFT
417|water area|shape=oval
418|car|color=blue||M
419|building|shape=cuboid|BOTTOM_RIGHT||roof
420|skyscraper|has_part=rectangular roof|BOTTOM|S
421|pavilion|shape=octagonal
422|football field|color=red
423|building|shape=L-shaped|TOP|S|roof
424|cross||TOP|S
"""


def quoted(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def target(category: str, attrs: dict[str, str]) -> str:
    args = [quoted(category)] + [f"{k}={quoted(v)}" for k, v in sorted(attrs.items())]
    return "T(" + ",".join(args) + ")"


def description(question: str) -> str:
    text = question.strip().rstrip("?.")
    # Only removal of reviewed interrogative wrappers; no new category inference.
    for prefix in (
        r"^What (?:main )?color (?:is|are) (?:there )?",
        r"^What are the colors of ",
        r"^What(?:'s| is) the (?:filling |background )?color of ",
        r"^What color ",
        r"^Where (?:is|are) ",
        r"^In which area of the picture is ",
        r"^In which area is ",
        r"^What area is ",
        r"^What is the location of ",
        r"^In the picture, where is ",
        r"^How many ",
    ):
        changed = re.sub(prefix, "", text, count=1, flags=re.I)
        if changed != text:
            text = changed
            break
    text = re.sub(r"\bare there\b", "", text, flags=re.I)
    text = re.sub(r"\s+(?:located )?in (?:the|this) (?:picture|diagram)$", "", text, flags=re.I)
    text = re.sub(r"\s+(?:located|shown)$", "", text, flags=re.I)
    return " ".join(text.split())


def specs() -> dict[int, tuple[str, dict[str, str], str, str, str]]:
    result = {}
    for line in REVIEWED.strip().splitlines():
        parts = (line.split("|") + [""] * 6)[:6]
        index, category, attributes, position, select, part = parts
        attrs = dict(s.split("=", 1) for s in attributes.split(";") if s)
        if int(index) in result:
            raise ValueError(f"duplicate reviewed row {index}")
        result[int(index)] = (category, attrs, position, select, part)
    # A reference described at an image position does not imply that the target
    # beside/above/outside it lies inside the same coarse image crop. These were
    # individually re-read; the complete reference position stays in SEM.query.
    for index in REFERENCE_REGION_CORRECTIONS:
        category, attrs, _position, select, part = result[index]
        result[index] = (category, attrs, "", select, part)
    return result


REFERENCE_REGION_CORRECTIONS = {
    7: "upper-right position attaches to the basketball-court reference",
    11: "upper-left building is a reference; surrounding parasols may extend outside its crop",
    14: "upper-right bridge/river is the reference for the nearby flag",
    21: "lower-right is relative to the larger ship stopping area, not the whole image",
    25: "upper-left grassland is the reference for the nearby windmill",
    30: "lower-left red ship is the reference for the adjacent buildings",
    31: "left-side ship is the reference for the adjacent truck",
    32: "left-side buildings are the references for the intervening flowers/trees",
    35: "upper-right red-roofed buildings are references for the coastal structure",
    39: "upper-right water is context for the nearby tennis court",
    45: "upper-left intersection is a reference; the car is above it",
    47: "upper-left intersection and crossing describe the reference context",
    49: "lower-left football field is the reference for the neighboring ride",
    52: "upper-right orange car is a reference for the neighboring car",
    55: "top-right roundabout is the reference for neighboring cars",
    59: "upper-edge sports stadium is the reference for the neighboring square",
    60: "bottom-right pale-blue area is the reference for the neighboring house",
    62: "upper-right zebra crossing is a reference for the neighboring car",
    67: "lower-left roundabout is the reference for the nearby crane",
    68: "upper-left circular water is a reference; surrounding buildings may cross the crop",
    73: "upper-left IMAX building is the reference for the container in front of its door",
    78: "central white roof is the reference for the neighboring boats",
    84: "lower-left basketball court is the reference for the nearby parked car",
    85: "upper-right crossing is the reference for the neighboring van",
    89: "bottom-left blue object is the reference for the neighboring car",
    92: "central white building is the reference for the neighboring board",
    95: "far-right building is the reference for nearby ships",
    96: "upper-right brown floor is the reference for the neighboring boat",
    101: "upper-right white building is the reference for the neighboring building",
    139: "far-left is relative to the water area, not the whole image",
    151: "lower-right court is the reference for the neighboring recreation facility",
    178: "left-side white barrel building is the reference; letters are above it",
    183: "right-side white building is the reference; pipeline is outside it",
    235: "upper-left bridge is the reference; ships are to its left",
    294: "central circular structures are the references; buildings are above them",
    360: "upper-left building is the reference; parking line is in front of it",
    374: "upper-right gray building is a reference; fencing is outside it",
}


def build_plan(index: int, sample: dict, spec: tuple) -> tuple[str, str]:
    category, attrs, position, selection, part = spec
    question = sample["question"]
    family = sample["metadata"]["source_category"]
    lines: list[str] = []

    def add(expression: str) -> str:
        ref = f"$n{len(lines) + 1}"
        lines.append(f"n{len(lines) + 1}={expression}")
        return ref

    scope = add(f"REGION($image0,{position})") if position else "$image0"
    t = target(category, attrs)
    residual = question
    # Singleton compound-object identity comes from intrinsic TargetSpec; only
    # external conditions use SELECT_SEM. Count has no extra LOCATE if unfiltered.
    if family == "count" and not selection:
        final = add(f"COUNT({scope},{t},{'false' if position else 'true'})")
        lines.append(f"FINAL({final},CHOICE_SINGLE)")
        return "\n".join(lines), "mme_count_intrinsic_region_v1"

    current = add(f"LOCATE({scope},{t})")
    mode = "SINGLE"
    if selection in {"S", "M"}:
        mode = "MULTI" if selection == "M" else "SINGLE"
        query = description(question)
        if index == 166:
            query = "yellow boats or yachts in the water area"
        if index == 231:
            query = "the divided lawn sections within the triangular square at the bottom right edge of the picture"
        if index == 396:
            query = "the tallest building"
        current = add(f"SELECT_SEM({current},null,{quoted(query)},{mode},{scope})")
    elif selection.startswith("E_"):
        current = add(f"SELECT_EXTREME({current},null,{selection[2:]})")
    elif selection in {"R_DESC", "R_ASC", "R_DESC_AROUND"}:
        order = "ASCENDING" if selection == "R_ASC" else "DESCENDING"
        current = add(f'SELECT_RANK({current},null,"bbox_area",1,{order})')
        if selection == "R_DESC_AROUND":
            current = add(f"SELECT_SUBREGION({current},null,AROUND)")
            mode = "VISUAL"
    elif selection in {"AROUND", "NEXT_TO", "INSIDE"}:
        subregion = "AROUND" if selection in {"AROUND", "NEXT_TO"} else "INSIDE"
        current = add(f"SELECT_SUBREGION({current},null,{subregion})")
        mode = "VISUAL"
    elif selection == "O_TOP":
        current = add(f"SELECT_ORD({current},null,1,TOP_TO_BOTTOM)")
    elif selection == "GROUP_CLUSTER":
        current = add(f"GROUP({current},CLUSTER)")
        mode = "VISUAL"
    elif selection == "ROW_ROAD":
        current = add(
            f'SELECT_SEM({current},null,"buildings on the right side of the road",MULTI,$image0)'
        )
        current = add(f"GROUP({current},ROW)")
        mode = "VISUAL"
    elif selection in {
        "TOP_ROAD",
        "LEFTMOST_ROAD",
        "LARGEST_ROAD",
        "LARGEST_RIVER",
        "LARGER_ADJACENT",
        "LARGEST_ROOF",
    }:
        query = {
            "TOP_ROAD": "cars on the road along the river"
            if index == 18
            else "cars on the highway",
            "LEFTMOST_ROAD": "cars on the slanted gray road",
            "LARGEST_ROAD": "trucks on the road below the center, with only one vehicle",
            "LARGEST_RIVER": "cargo ships in the river",
            "LARGER_ADJACENT": "cargo ships adjacent to each other in the lower right corner of the larger ship stopping area",
            "LARGEST_ROOF": "roofs of buildings in the area on the right edge of the road",
        }[selection]
        current = add(f"SELECT_SEM({current},null,{quoted(query)},MULTI,{scope})")
        if selection in {"TOP_ROAD", "LEFTMOST_ROAD"}:
            direction = "LEFTMOST" if selection == "LEFTMOST_ROAD" else "TOPMOST"
            current = add(f"SELECT_EXTREME({current},null,{direction})")
        else:
            current = add(f'SELECT_RANK({current},null,"bbox_area",1,DESCENDING)')
    elif selection == "DOCK_ORDINAL":
        # Explicit second dock row is selected before counting ships in it.
        lines.pop()
        docks = add('LOCATE($image0,T("dock"))')
        rows = add(f"GROUP({docks},COLUMN)")
        second = add(f"SELECT_ORD({rows},null,2,LEFT_TO_RIGHT)")
        current = add(f"SELECT_SUBREGION({second},null,INSIDE)")
        final = add(f"COUNT_IMAGE({current},{t},false)")
        lines.append(f"FINAL({final},CHOICE_SINGLE)")
        return "\n".join(lines), "mme_count_group_ordinal_v1"
    elif selection:
        raise ValueError(f"unsupported reviewed selection {selection}")

    if (
        family == "color"
        and re.match(r"^What (?:color are|are the colors of)\b", question, re.I)
        and not selection
        and index != 414
    ):
        mode = "MULTI"
    if family == "count":
        final = add(f"COUNT_ENTITIES({current},{t},false)")
        lines.append(f"FINAL({final},CHOICE_SINGLE)")
    elif family == "position" or mode in {"MULTI", "VISUAL"}:
        answer_type = "CHOICE_MULTI" if index in {139, 244} else "CHOICE_SINGLE"
        lines.append(f"FINAL_QUESTION({current},{answer_type},{quoted(residual)})")
    elif index == 166:
        lines.append(f"FINAL_QUESTION({current},CHOICE_SINGLE,{quoted(residual)})")
    else:
        args = [current, quoted("color")] + ([quoted(part)] if part else [])
        final = add("ATTRIBUTE(" + ",".join(args) + ")")
        lines.append(f"FINAL({final},CHOICE_SINGLE)")
    return "\n".join(lines), "mme_reviewed_target_" + (selection or "intrinsic") + "_v1"


def main() -> int:
    source_bytes = (ROOT / "mme_a.jsonl").read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != REVIEWED_SOURCE_SHA256:
        raise ValueError(
            "source shard differs from the questions reviewed for these row specifications"
        )
    samples = [json.loads(s) for s in source_bytes.decode("utf-8").splitlines()]
    reviewed = specs()
    if set(reviewed) != set(range(len(samples))):
        raise ValueError("reviewed row coverage does not match shard")
    output = ROOT / "plans" / "mme_a.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    written = 0
    counts: Counter = Counter()
    semantic_nodes = 0
    operator_counts: Counter = Counter()
    final_counts: Counter = Counter()
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for index, sample in enumerate(samples):
            try:
                dsl, rule = build_plan(index, sample, reviewed[index])
                graph = parse_taskgraph_dsl(dsl)
                _, report = validate_candidate(
                    graph,
                    inputs=sample["inputs"],
                    question=sample["question"],
                    question_type=sample["question_type"],
                )
                if not report.valid:
                    raise ValueError("; ".join(f"{e.code}:{e.message}" for e in report.errors))
                canonical = compile_taskgraph_to_dsl(graph)
                if canonicalize_target(parse_taskgraph_dsl(canonical)) != canonicalize_target(
                    graph
                ):
                    raise ValueError("canonical roundtrip mismatch")
                notes = "Source question reviewed by Codex; no image or ground-truth answer inspected. Structural checks do not establish visual correctness."
                if index in {144, 155, 220, 388, 405}:
                    notes += " Source wording is ambiguous or contains a translation typo; retain the original question and permit semantic unresolved outcomes rather than inventing visual facts."
                row = {
                    "sample_id": sample["sample_id"],
                    "dsl": canonical,
                    "rule_id": rule,
                    "review_notes": notes,
                }
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                counts[rule] += 1
                graph_dict = canonicalize_target(graph)
                operator_counts.update(n["op"] for n in graph_dict["nodes"])
                semantic_nodes += sum(
                    n["op"] == "SELECT" and n["params"].get("mode") == "SEMANTIC"
                    for n in graph_dict["nodes"]
                )
                final_counts[graph_dict["final"]["answer_type"]] += 1
                written += 1
            except Exception as exc:
                failures.append(
                    {
                        "index": index,
                        "sample_id": sample["sample_id"],
                        "question": sample["question"],
                        "error": str(exc),
                    }
                )
    (ROOT / "plans" / "mme_a_pending.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "written": written,
        "pending": failures,
        "rules": dict(counts),
        "operator_counts": dict(operator_counts),
        "semantic_node_count": semantic_nodes,
        "final_counts": dict(final_counts),
        "source_sha256": REVIEWED_SOURCE_SHA256,
        "review_method": "All 425 source questions read, per-row category/attributes/region/select specifications reviewed; no answer index, source answers, image loading or external API.",
        "checks": [
            "parse",
            "question-aware validate_candidate",
            "canonical JSON/DSL/JSON roundtrip",
        ],
        "corrected_initial_failures": {
            "count": 25,
            "reason": "Only SELECT_SEM permits a scope positional argument; removed invalid scope positions on the legacy SELECT modes.",
        },
        "second_review": {
            "reference_region_corrections": REFERENCE_REGION_CORRECTIONS,
            "group_operators_restored": [160, 177],
            "condition_loss_fixed": {
                "21": "preserve lower-right position relative to stopping area in semantic query",
                "83": "preserve fill part",
                "316": "preserve ground/background part",
                "396": "remove leftover interrogative wrapper without dropping tallest condition",
            },
            "tallest_not_bbox_area": [375, 396, 420],
            "intrinsic_external_boundary_fix": {
                "341": "white wake removed from intrinsic has_part; preserved in semantic query"
            },
        },
        "limitations": [
            "No model execution or visual answer validation",
            "A legal graph is not evidence of visual accuracy",
            "Ambiguous source text is flagged in per-row review notes",
        ],
    }
    (ROOT / "plans" / "mme_a_review.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    review_text = f"""# MME A 分片中文审核记录

当前分片共 {len(samples)} 条真实英文选择题，已逐条阅读问题并建立对应的类别、固有属性、区域和选择规则；输出 {written} 条，待处理 {len(failures)} 条。生成器只应用这些经审阅的逐行规则，不以关键词猜测未审阅题目的 gold。没有读取答案索引、GT、图片或调用外部模型 API。

源文件 SHA256 为 {REVIEWED_SOURCE_SHA256}，脚本在生成前强制核对字节哈希，源题顺序或内容改变即拒绝沿用旧标注。

## 检查结果

全部输出逐条通过 DSL 解析、带原问题和 inputs 的 validate_candidate，以及规范化 JSON → DSL → JSON 往返。Final 类型为 {dict(final_counts)}；COUNT 节点 {operator_counts.get("COUNT", 0)}，SELECT_SEM 节点 {semantic_nodes}，GROUP 节点 {operator_counts.get("GROUP", 0)}。未添加 BBOX、VLM_REASON 或预测答案坐标。

## 本次二次语义复核

- 修正 {len(REFERENCE_REGION_CORRECTIONS)} 个参考物位置被错误提升为目标整图裁剪的情况。参考物角落、外部邻接和相对参考区域方向保留在完整 SELECT_SEM query 中；只有明确属于目标自身的图像区域继续使用 REGION。
- 第 21 题保留“较大停泊区域的右下角”，没有将其改写为整图右下角；第 139 题保留水域自身的最左侧；第 294 题没有将“位于中心参考物上方”的建筑先裁进 CENTER。
- 明确的最大/最小面积、最左/最右/最上仍使用 RANK/EXTREME。第 2 题定位 roof 候选并按屋顶 bbox_area 排序，没有把建筑面积冒充屋顶面积。top car 先筛道路条件再执行 TOPMOST。
- 第 160 题恢复 GROUP ROW，第 177 题恢复 GROUP CLUSTER；第 353 题按“从左第二排码头”使用 COLUMN 分组、LEFT_TO_RIGHT 序数、子区域和 COUNT，避免由 SEM 代替明确的分组或顺序。
- 第 375、396、420 题的 tallest 不等于 bbox_area；当前 RANK 没有 height 标准，因此完整保留于 SELECT_SEM。第 388 题 first car 缺少排序方向，保留完整语义条件并标记歧义，没有编造 TOP_TO_BOTTOM。
- 第 83 题保留 fill，第 316 题保留 ground/background。白色航迹属于周边关联条件，第 341 题已从固有 has_part 中移出并交由 SELECT_SEM 消费。
- 多目标颜色使用视觉集合和静态原问题 FINAL_QUESTION，未将 MULTI 选择直接交给 ATTRIBUTE。单个建筑复合体的 exterior walls 可以作为 ATTRIBUTE 的 part，避免仅因表层 are 而错误判成多个目标。

## 限制

结构合法不代表真实图片上答案正确；本次没有执行视觉推理或模型准确率实验。源题含翻译错误或歧义的第 144、155、220、388、405 题已写入逐行 review_notes，执行时可能得到未决结果。此审核只覆盖 mme_a 分片，不修改其他分片、旧训练集或固定测试集。
"""
    (ROOT / "plans" / "mme_a_review.md").write_text(review_text, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
