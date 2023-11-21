# Compare the two files
# use numpy tofile function to dump the data
# ndarray.tofile("output1_...", sep=",")
index = 1
name1 = "output1_" + str(index) + "_{}.txt"
name2 = "output2_" + str(index) + "_{}.txt"

for n in [
    "image",
    "joint",
    "y_image",
    "y_joint",
    "ect_pts",
    "dec_pts",
    "state0",
    "state1",
    "in_state0",
    "in_state1",
]:
    with open(name1.format(n), "r") as f1, open(name2.format(n), "r") as f2:
        l1 = f1.readline()
        l2 = f2.readline()
        sep = ","
        l1 = l1.split(sep)
        l2 = l2.split(sep)
        l1 = [float(i) for i in l1]
        l2 = [float(i) for i in l2]

        # diff
        print("Find diff in: ", n)
        if len(l1) != len(l2):
            print("len diff: ", n, len(l1), len(l2))
            break
        for i in range(len(l1)):
            if abs(l1[i] - l2[i]) > 0.0001:
                print("diff: ", n, i, l1[i], l2[i])
                break
        print("====================================")
