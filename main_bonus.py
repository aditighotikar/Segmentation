import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.data import astronaut
from skimage.util import img_as_float
import maxflow
from scipy.spatial import Delaunay

BLUE = [255, 0, 0]  # rectangle color
RED = [0, 0, 255]  # PR BG
GREEN = [0, 255, 0]  # PR FG
BLACK = [0, 0, 0]  # sure BG
WHITE = [255, 255, 255]  # sure FG

DRAW_BG = {'color': BLUE}
DRAW_FG = {'color': RED}

# setting up flags
rect = (0, 0, 1, 1)
drawing = False  # flag for drawing curves
value = DRAW_FG  # drawing initialized to FG
thickness = 4  # brush thickness

def help_message():
   print("Usage: [Input_Image] ")
   print("[Input_Image]")
   print("Path to the input image")
   print(sys.argv[0] + " astronaut.png ")

# Calculate the SLIC superpixels, their histograms and neighbors
def superpixels_histograms_neighbors(img):
    # SLIC
    segments = slic(img, n_segments=500, compactness=18.48)
    segments_ids = np.unique(segments)

    # centers
    centers = np.array([np.mean(np.nonzero(segments == i), axis=1) for i in segments_ids])

    # H-S histograms for all superpixels
    hsv = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2HSV)
    bins = [20, 20]  # H = S = 20
    ranges = [0, 360, 0, 1]  # H: [0, 360], S: [0, 1]
    colors_hists = np.float32(
        [cv2.calcHist([hsv], [0, 1], np.uint8(segments == i), bins, ranges).flatten() for i in segments_ids])

    # neighbors via Delaunay tesselation
    tri = Delaunay(centers)

    return (centers, colors_hists, segments, tri.vertex_neighbor_vertices)


# Get superpixels IDs for FG and BG from marking
def find_superpixels_under_marking(marking, superpixels):
    fg_segments = np.unique(superpixels[marking[:, :, 0] != 255])
    bg_segments = np.unique(superpixels[marking[:, :, 2] != 255])
    return (fg_segments, bg_segments)


# Sum up the histograms for a given selection of superpixel IDs, normalize
def cumulative_histogram_for_superpixels(ids, histograms):
    h = np.sum(histograms[ids], axis=0)
    return h / h.sum()


# Get a bool mask of the pixels for a given selection of superpixel IDs
def pixels_for_segment_selection(superpixels_labels, selection):
    pixels_mask = np.where(np.isin(superpixels_labels, selection), True, False)
    return pixels_mask


# Get a normalized version of the given histograms (divide by sum)
def normalize_histograms(histograms):
    return np.float32([h / h.sum() for h in histograms])


# Perform graph cut using superpixels histograms
def do_graph_cut(fgbg_hists, fgbg_superpixels, norm_hists, neighbors):
    num_nodes = norm_hists.shape[0]
    # Create a graph of N nodes, and estimate of 5 edges per node
    g = maxflow.Graph[float](num_nodes, num_nodes * 5)
    # Add N nodes
    nodes = g.add_nodes(num_nodes)

    hist_comp_alg = cv2.HISTCMP_KL_DIV

    # Smoothness term: cost between neighbors
    indptr, indices = neighbors
    for i in range(len(indptr) - 1):
        N = indices[indptr[i]:indptr[i + 1]]  # list of neighbor superpixels
        hi = norm_hists[i]  # histogram for center
        for n in N:
            if (n < 0) or (n > num_nodes):
                continue
            # Create two edges (forwards and backwards) with capacities based on
            # histogram matching
            hn = norm_hists[n]  # histogram for neighbor
            g.add_edge(nodes[i], nodes[n], 20 - cv2.compareHist(hi, hn, hist_comp_alg),
                       20 - cv2.compareHist(hn, hi, hist_comp_alg))

    # Match term: cost to FG/BG
    for i, h in enumerate(norm_hists):
        if i in fgbg_superpixels[0]:
            g.add_tedge(nodes[i], 0, 1000)  # FG - set high cost to BG
        elif i in fgbg_superpixels[1]:
            g.add_tedge(nodes[i], 1000, 0)  # BG - set high cost to FG
        else:
            g.add_tedge(nodes[i], cv2.compareHist(fgbg_hists[0], h, hist_comp_alg),
                        cv2.compareHist(fgbg_hists[1], h, hist_comp_alg))

    g.maxflow()
    return g.get_grid_segments(nodes)


def onmouse(event, x, y, flags, param):
    global img, img_backup, drawing, value, mask, rectangle, rect, rect_or_mask, ix, iy, rect_over

    #On mouse events
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(img, (x, y), thickness, value['color'], -1)
        cv2.circle(mask, (x, y), thickness, value['color'], -1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img, (x, y), thickness, value['color'], -1)
            cv2.circle(mask, (x, y), thickness, value['color'], -1)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing == True:
            drawing = False
            cv2.circle(img, (x, y), thickness, value['color'], -1)
            cv2.circle(mask, (x, y), thickness, value['color'], -1)


if __name__ == '__main__':

    # validate the input arguments
    if len(sys.argv) == 2:
        filename = sys.argv[1]  
    else:
        help_message()
		
    global img, mask, output
    img = cv2.imread(filename)
    img_backup = img.copy()  
    mask = 255 * np.ones(img.shape, dtype=np.uint8)
    output = np.zeros(img.shape, dtype=np.uint8)  # output image to be shown

    # input and output windows
    cv2.namedWindow('output')
    cv2.namedWindow('input')
    cv2.setMouseCallback('input', onmouse)
    cv2.moveWindow('input', img.shape[1] + 90, 200)
    
	#Print instructions
    print("Step 1: Press 0. Then, press left mouse button and move cursor to draw background markings.")
    print("Step 2: Press 1. Then, press left mouse button and move cursor to draw foreground markings.")
    print("Step 3: Press 'c' to confirm/render segmentation.")
    print("Step 4: Press 'r' to reset. Repeat steps 1-3.")
    print("Step 5: Press Esc to exit.")
	
    #Calculate the segments/superpixels
    centers, color_hists, superpixels, neighbors = superpixels_histograms_neighbors(img)

    while (1):

        cv2.imshow('output', output)
        cv2.imshow('input', img)
        key = cv2.waitKey(1)

        # key bindings
        if key == 27:  # esc to exit
            break
		#Background drawing
        elif key == ord('0'):  
            value = DRAW_BG
		#Foreground drawing
        elif key == ord('1'):  
            value = DRAW_FG
        #Resetting 
        elif key == ord('r'):  
            rect = (0, 0, 1, 1)
            drawing = False
            value = DRAW_FG
            img = img_backup.copy()
			#Create the mask
            mask = 255 * np.ones(img.shape, dtype=np.uint8)
            output = np.zeros(img.shape, dtype=np.uint8)  
        elif key == ord('c'):  
            # Find the fb and bg segments and combine them to a numpy array
            fg_segments, bg_segments = find_superpixels_under_marking(mask, superpixels)
            fgbg_superpixels = (fg_segments, bg_segments)

            # Find the fb and bg histograms and combine them to a numpy array
            fg_cumulative_hist = cumulative_histogram_for_superpixels(fg_segments, color_hists)
            bg_cumulative_hist = cumulative_histogram_for_superpixels(bg_segments, color_hists)
            fgbg_hists = (fg_cumulative_hist, bg_cumulative_hist)

            norm_hists = normalize_histograms(color_hists)

            # Perform the graph-cut algorithm
            graph_cut = do_graph_cut(fgbg_hists, fgbg_superpixels, norm_hists, neighbors)

            output = pixels_for_segment_selection(superpixels, np.nonzero(graph_cut))
            output = np.uint8(output * 255)

cv2.destroyAllWindows()