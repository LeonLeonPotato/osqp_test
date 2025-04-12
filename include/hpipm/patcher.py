import os

root = "/Users/leon.zhu/Desktop/projects/blasfeo-test/include/"

def process(lines):
    for i, line in enumerate(lines):
        if not line.startswith("#include"): continue
        if "<blasfeo" in line or "<hpipm" in line:
            lines[i] = lines[i].replace("<blasfeo", "\"blasfeo")
            lines[i] = lines[i].replace("\"blasfeo", "\"blasfeo/blasfeo")
            lines[i] = lines[i].replace("<", "\"").replace(">", "\"")

for file in os.listdir(os.path.join(root, "hpipm")):
    if file.endswith(".py"): continue
    with open(os.path.join(root, "hpipm", file), "r") as f:
        lines = f.readlines()
    process(lines)
    with open(os.path.join(root, "hpipm", file), "w") as f:
        f.writelines(lines)
