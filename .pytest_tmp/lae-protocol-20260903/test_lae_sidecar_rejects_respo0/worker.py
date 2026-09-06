import json, sys
for line in sys.stdin:
    json.loads(line)
    sys.stdout.write('{"id":"wrong-id","status":"ok","bbox_list":[],"bbox_scores":[]}' + '\n'); sys.stdout.flush()
