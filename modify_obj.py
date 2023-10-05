

vt_count = 0
modified = open("obj_models/model_modified.obj", 'w')

with open("obj_models/model.obj", "r") as file:
    while line := file.readline():
        l = line.rstrip()
        split = l.split()
        if(len(split) > 1 and (split[0] == 'vt')):
            continue
        else:
            modified.write(l + "\n")

