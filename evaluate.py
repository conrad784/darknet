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
        # we might have negative width and height, lets standardize
        if self.width < 0:
            self.min_x = self.min_x - self.width
            self.width = abs(self.width)
        if self.height < 0:
            self.min_y = self.min_y - self.height
            self.height = abs(self.height)

    @property
    def max_x(self):
        return self.min_x+self.width

    @property
    def max_y(self):
        return self.min_y+self.height

    @property
    def area(self):
        return self.width * self.height

    def intersects(self, other):
        if self.min_x > other.max_x or self.max_x < other.min_x:
            return False
        if self.min_y > other.max_y or self.max_y < other.min_y:
            return False
        return True

    def get_overlap(self, rect):
        # calculate overlap
        intersect_region = self.union(rect)
        overlap_area = intersect_region.area
        return overlap_area

    def union(self, other):
        if not self.intersects(other):
            return
        min_x = max(self.min_x, other.min_x)
        max_x = min(self.max_x, other.max_x)
        min_y = max(self.min_y, other.min_y)
        max_y = min(self.max_y, other.max_y)
        width = max_x - min_x
        height = max_y - min_y

        return Rectangle((min_x, max_x, width, height))

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

    def get_metrics(self):
        if not self.ground_truth:
            print("[ERROR] no ground truth available")
            return
        self.intersections = []
        self.precision = []
        self.recall = []
        self.iou = []

        for i, box in enumerate(self.found_boxes):
            box = Rectangle(box)
            # there seems to be the issue of bounding boxes with either width or height = 0
            if not box.area:
                continue
            for j, region in enumerate(self.ground_truth):
                rect = Rectangle(region)
                if rect.intersects(box):
                    # get iou stuff
                    if not box.area:
                        import pdb
                        pdb.set_trace()
                    overlap = rect.get_overlap(box)
                    iou = overlap / (box.area+rect.area-overlap)
                    if args.verbose > 4:
                        print(f"[DEBUG] iou ({i},{j}): {iou}")
                    # test if better than already found intersection
                    idx = -1
                    for i, p in enumerate(self.intersections):
                        if (i, j) == p:
                            if iou > self.iou[i]:
                                idx = i
                    if idx > -1:
                        self.intersections[idx] = (i,j)
                        self.iou[idx] = iou
                    else:
                        self.intersections.append((iou, (i, j)))
                        self.iou.append(iou)

                    # get precision
                    precision = overlap / box.area
                    self.precision.append((precision, (i, j)))
                    # get recall
                    recall = overlap / rect.area
                    self.recall.append((recall,  (i, j)))


        # sanity check
        if len(self.intersections) > len(self.ground_truth):
            # this usually happens for very bad predictions, e.g. batch iteration < 500
            if args.verbose > 4:
                print(f"[ERROR] too many intersections found: {self.name}")

    def get_accuracy(self, text=False):
        try:
            _ = self.intersections
        except AttributeError:
            self.get_metrics()
        if text:
            return f"{len(self.intersections)}/{len(self.ground_truth)}"
        else:
            ret = len(self.intersections)/len(self.ground_truth)
            # little sanity check
            if ret <= 1:
                return ret
            else:
                return 0

    def get_mean_iou(self):
        try:
            _ = self.intersections
        except AttributeError:
            self.get_metrics()

        ious = [x[0] for x in self.intersections]
        if ious:
            return sum(ious)/len(ious)
        else:
            return 0

    def get_precision(self):
        try:
            _ = self.precision
        except AttributeError:
            self.get_metrics()
        return self.precision

    def get_recall(self):
        try:
            _ = self.recall
        except AttributeError:
            self.get_metrics()
        return self.recall

# utility functions
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
    import numpy as np

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
    parser.add_argument('--only-plot', action='store_true',
                        help='only replot the data')

    args = parser.parse_args()

    glyph_storage = defaultdict(list)

    if not args.only_plot:
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

        print("[INFO] now evaluating result files and creating images")

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
                    # we got a labeled image here, lets calculate some metrics
                    myimg = MayaImg(key)
                    myimg.ground_truth = ground_truth
                    myimg.found_boxes = boxes_sane
                    glyph_storage[weight].append(myimg)

        with open("glyph_storage.pickle", "wb") as fd:
            pickle.dump(glyph_storage, fd)

    # only evaluation from here on
    if not glyph_storage: # load old pickle, because we only want to evaluate and not generate new
        with open("glyph_storage.pickle", "rb") as fd:
            glyph_storage = pickle.load(fd)

    weight_statistic = defaultdict(dict)
    for w, data in glyph_storage.items():
        lweight = int(w.split("_")[-1].split(".")[0])
        weight_statistic[lweight]["img"] = []
        weight_statistic[lweight]["iou"] = []
        weight_statistic[lweight]["accuracy"] = []
        for im in data:
            if im.ground_truth:  # only evaluate if ground truth available
                weight_statistic[lweight]["img"].append(im)

                ac = im.get_accuracy()
                mean_iou = im.get_mean_iou()
                weight_statistic[lweight]["iou"].append(mean_iou)
                weight_statistic[lweight]["accuracy"].append(ac)

    iou_over_batches = []
    accuracy_over_batches = []
    for w in sorted(weight_statistic.keys()):
        iou = np.array(weight_statistic[w].get("iou")).mean()
        acc = np.array(weight_statistic[w].get("accuracy")).mean()
        iou_over_batches.append([w, iou])
        accuracy_over_batches.append([w, acc])
        print(f"{w} accuracy: {acc}")

    iou_over_batches = np.array(iou_over_batches)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(iou_over_batches[:,0], iou_over_batches[:,1])
    ax.axhline(y=0.5, color='red', label='IoU limit')
    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)
    ax.set_xlabel("Batch number")
    ax.set_ylabel("IoU")
    ax.legend()
    ax.set_title("IoU over training batches")
    fig.savefig("plots/iou_over_batches.png")

    accuracy_over_batches = np.array(accuracy_over_batches)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(accuracy_over_batches[:,0], accuracy_over_batches[:,1])
    ax.axhline(y=accuracy_over_batches[:,1].mean(), label='mean accuracy')
    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)
    ax.set_xlabel("Batch number")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.set_title("Accuracy over training batches")
    fig.savefig("plots/accuracy_over_batches.png")

    best_batch_iou, best_iou = iou_over_batches[np.argmax(iou_over_batches[:,1])]
    print(f"[INFO] best batch was {best_batch_iou} with IoU: {best_iou}")

    best_batch_acc, best_acc = accuracy_over_batches[np.argmax(accuracy_over_batches[:,1])]
    print(f"[INFO] best batch was {best_batch_acc} with accuracy: {best_acc}")

    interest_batch = glyph_storage[f"yolo-maya_{int(best_batch_acc)}.weights"]

    print(f"Evaluating for batch {best_batch_acc}")
    for im in interest_batch:
        print(f"{im.name}")
        print("found {}".format(im.get_accuracy(text=True)))
        print("Recall:", np.array([x[0] for x in im.recall]).mean())
        print("Precision:", np.array([x[0] for x in im.precision]).mean())
