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
class Rectangle():
    def __init__(self, region):
        x, y, width, heigth = region
        self.min_x = int(x)
        self.min_y = int(y)
        self.width = int(width)
        self.height = int(heigth)
        self.max_x = self.min_x+self.width
        self.max_y = self.min_y+self.height

    def intersects(self, rect, min_overlap=0.5):
        if self.min_x > rect.max_x or self.max_x < rect.min_x:
            return False
        if self.min_y > rect.max_y or self.max_y < rect.min_x:
            return False
        # calculate overlap
        area = self.width * self.height
        rect_area = rect.width * rect.height
        if rect_area > area*1.3:  # test if rectangle is very big
            if args.verbose:
                print("[ERROR] seems like the found bounding boxes are too big")
            return False
        intersect_region = self.get_intersect_region(rect)
        overlap_area = (intersect_region.max_x - intersect_region.min_x) \
            * (intersect_region.max_y - intersect_region.min_y)
        if overlap_area > area*min_overlap:
            return True

    def get_intersect_region(self, other):
        min_x = max(self.min_x, other.min_x)
        max_x = min(self.max_x, other.max_x)
        min_y = max(self.min_y, other.min_y)
        max_y = min(self.max_y, other.max_y)

        return Rectangle((min_x, max_x, min_y, max_y))

    def __repr__(self):
        return f"<Rectangle>{self.min_x}, {self.min_y}, {self.width}, {self.height}"

    def __str__(self):
        return f"{self.min_x}, {self.min_y}, {self.width}, {self.height}"


class MayaImg():
    def __init__(self, path):
        self.path = path
        self.name = path.split("/")[-1]
        self.ground_truth = None
        self.found_boxes = None

    def __repr__(self):
        return f"<MayaImg>{self.path}"

    def __str__(self):
        return f"{self.path}"

    def get_iou(self):
        if not self.ground_truth:
            print("[ERROR] no ground truth available")
            return
        self.intersections = []
        for i, region in enumerate(self.ground_truth):
            region = Rectangle(region)
            for j, box in enumerate(self.found_boxes):
                rect = Rectangle(box)
                if region.intersects(rect):
                    self.intersections.append((i, j))
        # sanity check
        if len(self.intersections) > len(self.ground_truth):
            print(f"[ERROR] too many intersections: {self.intersections}")

    def get_accuracy(self):
        try:
            _ = self.intersections
        except AttributeError:
            self.get_iou()
        return f"{len(self.intersections)}/{len(self.ground_truth)}"


def create_rect(t, color='red'):
    if type(t) == str:
        _, t = convert_yolo_string(t)
    left_x, top_y, width, height = t
    return matplotlib.patches.Rectangle((int(left_x), int(top_y)), int(width), int(height),
                                        fill=False, edgecolor=color, linewidth=2)


def draw_bboxes(img, regions, ground_truth=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    for prop in regions:
        ax.add_patch(create_rect(prop))
    if ground_truth:
        for prop in ground_truth:
            ax.add_patch(create_rect(prop, 'green'))
    return fig, ax


def convert_yolo_string(s):
    decode = s.strip().replace("(", "").replace(")", "").split()
    _, left_x, _, top_y, _, width, _, height = decode
    confidence = s.split(" ")[1].replace("%", "")
    return confidence, (left_x, top_y, width, height)


def convert_yolo_coordinates(x_c_n, y_c_n, width_n, height_n, img_width, img_height):
    # remove normalization given the size of the image
    x_c = float(x_c_n) * img_width
    y_c = float(y_c_n) * img_height
    width = float(width_n) * img_width
    height = float(height_n) * img_height
    # compute half width and half height
    half_width = width / 2
    half_height = height / 2
    # compute left, top, right, bottom
    left = int(x_c - half_width)
    top = int(y_c - half_height)
    right = int(x_c + half_width)
    bottom = int(y_c + half_height)
    box_height = bottom-top
    box_width = right-left
    return left, top, box_width, box_height


def get_ground_truth(ground_truth_file):
    try:
        with open(ground_truth_file, "r") as fd:
            tmp_ground_truth = fd.readlines()
    except FileNotFoundError:
        if args.verbose:
            print(f"[INFO] no ground truth for {img_path}, just predicting")
        return None

    ground_truth = []
    img = skimage.io.imread(img_path)
    img_height, img_width = img.shape[:2]
    for box in tmp_ground_truth:
        t, x_c_n, y_c_n, width_n, height_n = box.strip().split(" ")
        b = convert_yolo_coordinates(x_c_n, y_c_n, width_n,
                                     height_n, img_width, img_height)

        ground_truth.append(b)

    return ground_truth

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
    import skimage.io
    from tqdm import tqdm
    from collections import defaultdict
    import pickle


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
    parser.add_argument('-s', '--storefolder', default="unkown",
                        help='specify folder where pictures get stored')
    parser.add_argument('-gt', '--ground-truth', action='store_true',
                        help='draw ground-truth on image')

    args = parser.parse_args()

    glyph_storage = defaultdict(list)

    weights = os.listdir(args.weights[0])
    results = []
    for weight in weights:
        eval_type = args.images[0].split("-")[-1].split(".")[0]
        result_file = f"results/result-{eval_type}_{weight}.txt"
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

    if not os.path.exists(f"evaluation/{args.storefolder}"):
        os.mkdir(f"evaluation/{args.storefolder}")
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
            boxes_sane = [convert_yolo_string(box)[1] for box in boxes]
            img_path = key
            ground_truth_file = f"{img_path[:-4]}.txt"
            ground_truth = get_ground_truth(ground_truth_file)

            eval_file = f"evaluation/{args.storefolder}/{key.split('/')[-1]}_bboxes_{weight}.png"
            if os.path.isfile(eval_file) and not args.purge:
                print(f"[INFO] {eval_file} already exists, skipping...")
            else:
                try:
                    img = skimage.io.imread(img_path)
                    fig, ax = draw_bboxes(img, boxes, ground_truth)
                    fig.savefig(eval_file)
                    plt.close()
                except Exception as e:
                    print(f"[ERROR] {e}")

            if ground_truth:
                # we got a labeled image here, lets calculate accuracy
                myimg = MayaImg(key)
                myimg.ground_truth = ground_truth
                myimg.found_boxes = boxes_sane
                glyph_storage[weight].append(myimg)

    with open("glyph_storage.pickle", "wb") as fd:
        pickle.dump(glyph_storage, fd)

    for w, data in glyph_storage.items():
        for im in data:
            if im.ground_truth:  # only evaluate if there is actual ground-truth
                ac = im.get_accuracy()
                print(f"{w} with accuracy {ac}")
