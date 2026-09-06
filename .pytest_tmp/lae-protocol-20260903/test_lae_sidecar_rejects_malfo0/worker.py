import json, sys
for line in sys.stdin:
    json.loads(line)
    sys.stdout.write("'not-json'" + '\n'); sys.stdout.flush()
