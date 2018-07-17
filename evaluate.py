#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (C) 2018 Conrad Sachweh

"""NAME
        %(prog)s - <description>

SYNOPSIS
        %(prog)s [--help]

DESCRIPTION
        none

FILES
        none

SEE ALSO
        nothing

DIAGNOSTICS
        none

BUGS
        none

AUTHOR
        Conrad Sachweh, conrad@csachweh.de
"""
DARKNET_BIN = "darknet-nolabel"
#--------- Classes, Functions, etc ---------------------------------------------
def create_rect(dn_str):
    decode = dn_str.strip().replace("(", "").replace(")", "").split()
    _, left_x, _, top_y, _, width, _, height = decode
    return matplotlib.patches.Rectangle((int(left_x), int(top_y)), int(width), int(height),
                                        fill=False, edgecolor='red', linewidth=2)


def draw_bboxes(img, regions):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    for prop in regions:
        ax.add_patch(create_rect(prop))
    return fig, ax
#-------------------------------------------------------------------------------
#    Main
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import os
    import argparse
    import subprocess
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--quiet', action='store_true', dest='quiet',
                        help="don't print status messages to stdout")
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='show more verbose output')

    parser.add_argument('--purge', action='store_true',
                        help="don't skip already existing files")
    parser.add_argument('-w', '--weights', nargs=1, default="backup/",
                        help='weights folder')
    parser.add_argument('images', nargs=1,
                        help='images file (same as for darknet itself)')

    args = parser.parse_args()

    weights = os.listdir(args.weights[0])
    results = []
    for weight in weights:
        result_file = f"results/result_{weight}.txt"
        if os.path.isfile(result_file) and not args.purge:
            print(f"[INFO] {result_file} already exists, skipping...")
        else:
            CMD = [f"./{DARKNET_BIN}", "detector", "test", "data/maya.data",
                   "cfg/yolo-maya-testing.cfg", f"{args.weights[0]}{weight}", "-thresh 0.1",
                   "-dont_show", "-ext_output", "<", args.images[0], ">", result_file]
            print(f"[INFO] evaluating {weight} ...")
            if args.verbose:
                print(f"[DEBUG] calling: {CMD}")
            ret = subprocess.run(" ".join(CMD), stderr=subprocess.PIPE, check=True, shell=True)
            if ret.returncode:
                print(f"[ERROR] stderr ended with: {ret.stderr[-200:]}")
                break
        with open(result_file, "r") as fd:
            results.append(fd.readlines())

    print("[INFO] now evaluating result files")

    from collections import defaultdict
    import skimage.io
    for weight, result in tqdm(zip(weights, results)):
        glyphs = defaultdict(list)
        # we got a list of text here
        for i in range(len(result)):
            if result[i].startswith("Enter Image Path:"):
                # now there are glyphs
                path = result[i].split(" ")[3].replace(":", "")
                for j in range(i+1, len(result)):
                    if result[j].startswith("glyph"):
                        g = result[j].split("(")[1]
                        glyphs[path].append(g)
                    else:
                        # no glyphs anymore
                        i = j
                        break

        for key, boxes in tqdm(glyphs.items()):
            eval_file = f"evaluation/{key.split('/')[-1]}_bboxes_{weight}.png"
            if os.path.isfile(eval_file) and not args.purge:
                print(f"[INFO] {eval_file} already exists, skipping...")
            else:
                img = skimage.io.imread(key)
                try:
                    fig, ax = draw_bboxes(img, boxes)
                    fig.savefig(eval_file)
                    plt.close()
                except Exception as e:
                    print(f"[ERROR] {e}")
