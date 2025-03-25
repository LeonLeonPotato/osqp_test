import os

for file in os.listdir("./include/OsqpEigen"):
    if file.endswith("py"): continue
    with open("./include/OsqpEigen/" + file, "r") as f:
        lines = f.readlines()

    for i in range(len(lines)):
        line = lines[i]
        if not ("#include" in line): continue
        if "osqp" in line.lower():
            line = line.replace("<", "\"")
            line = line.replace(">", "\"")
            lines[i] = line
        
    with open("./include/OsqpEigen/" + file, "w") as f:
        f.writelines(lines)