import argparse as ap
import glob, os

def main():

    parser = ap.ArgumentParser(description="Script to combine scan files into a single scan in cartesian form")
    parser.add_argument("scan_folder_name", type=str)
    parser.add_argument("--output_file", type=str, default="combined.csv")

    args, leftovers = parser.parse_known_args()

    coords = []
    for file in glob.glob("./"+args.scan_folder_name+"/*"):
    	coords.extend(read_file(filename))

if __name__=="__main__":
	main()