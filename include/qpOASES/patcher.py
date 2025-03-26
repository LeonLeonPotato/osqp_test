import os

def process(handle):
    lines = handle.readlines()
    for i, line in enumerate(lines):
        if "<qpOASES" in line:
            lines[i] = line.replace("<", "\"").replace(">", "\"")
    return ''.join(lines)

for path, _, file in os.walk('include/qpOASES'):
    for f in file:
        if f.endswith('p'):
            p = open(os.path.join(path, f))
            processed = process(p)
            p.close()
            with open(os.path.join(path, f), 'w') as p:
                p.write(processed)