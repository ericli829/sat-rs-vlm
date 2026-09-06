import json, sys
for line in sys.stdin:
    json.loads(line)
    sys.stdout.write('{"status":"failed","failure_stage":"model_init","error":"missing weights"}' + '\n'); sys.stdout.flush()
