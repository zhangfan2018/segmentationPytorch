
import numpy as np

from skimage import measure
from skimage.morphology import label


mask = np.random.randint(2, size=(256, 192, 256))

mask[mask != 0] = 1
labeled_mask, num = label(mask, neighbors=8, background=0, return_num=True)
region_props = measure.regionprops(labeled_mask)

for i in range(num):
    props = region_props[i]
    bbox = props.bbox
    centroid = props.centroid
    pass