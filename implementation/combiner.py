import argparse as ap
import glob
import matplotlib.pyplot as plt
from scanreader import Scan
from tqdm import tqdm
import numpy as np


def read_file(filename):
    scan = Scan(filename, absolute=True)
    return scan.scan_points


def main():
    parser = ap.ArgumentParser(description="Script to combine scan files into a single scan in cartesian form")
    parser.add_argument("scan_folder_name", type=str)
    parser.add_argument("--output_file", type=str, default="combined.csv")
    parser.add_argument("--draw", action='store_true')

    args, leftovers = parser.parse_known_args()

    files = glob.glob("./" + args.scan_folder_name + "/*")
    files.sort()
    arrx = []
    arry = []
    i = 0
    with open(args.output_file, 'w+') as fd:
        for f in tqdm(files):
            i += 1
            points = read_file(f)
            points = [x.tolist() for x in points]
            for point in points:
                arrx.append(point[0])
                arry.append(point[1])
                fd.write(",".join(map(str, point)) + "\n")

    if args.draw:
        data = np.genfromtxt(args.output_file, delimiter=',', names=['x', 'y'])
        plt.figure("X/Y")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.scatter(data['x'], data['y'], s=2, marker='x')
        plt.show()


if __name__ == "__main__":
    main()
