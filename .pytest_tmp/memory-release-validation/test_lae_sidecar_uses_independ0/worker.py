import json, sys, time
for line in sys.stdin:
    request = json.loads(line)
    time.sleep(0.03)
    response = {'id': request['id'], 'status': 'ok', 'bbox_list': [], 'bbox_scores': [], 'metadata': {'image_width': 20, 'image_height': 10}}
    sys.stdout.write(json.dumps(response) + '\n')
    sys.stdout.flush()
