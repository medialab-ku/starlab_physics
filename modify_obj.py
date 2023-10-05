

vt_count = 0
modified = open("obj_models/kyra_model_modified.obj", 'w')

with open("obj_models/kyra_model.obj", "r") as file:
    while line := file.readline():
        l = line.rstrip()
        split = l.split()
        if(len(split) > 1 and (split[0] == 'vt' or split[0] == 'vn')):
            continue
        else:
            modified.write(l + "\n")

