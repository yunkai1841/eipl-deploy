import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json


def main(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    x = [d["name"] for d in data[1:]]
    # add \n to long name
    for i, name in enumerate(x):
        if len(name) > 30:
            x[i] = name[:30] + "\n" + name[30:]
    y = [d["averageMs"] for d in data[1:]]
    axis_name = ("name", "averageMs")
    plt.rcParams["font.size"] = 5
    plt.rcParams["figure.subplot.bottom"] = 0.5
    plt.figure(figsize=(12, 4))

    plt.bar(x, y)
    plt.xticks(rotation=90)

    plt.xlabel(axis_name[0])

    plt.ylabel(axis_name[1])
    plt.savefig("profile.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="profile_cuda.json")
    args = parser.parse_args()
    main(args.path)
