import numpy as np
import logging

import sys
import mrcfile
args = sys.argv
import numpy as np
import os
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import resize

def maxmask(tomo, side=5,percentile=60):
    from scipy.ndimage.filters import maximum_filter
    # print('maximum_filter')
    filtered = maximum_filter(-tomo, 2*side+1, mode='reflect')
    out =  filtered > np.percentile(filtered,100-percentile)
    out = out.astype(np.uint8)
    return out

def stdmask(tomo,side=10,threshold=60):
    from scipy.signal import convolve
    # print('std_filter')
    tomosq = tomo**2
    ones = np.ones(tomo.shape)
    eps = 0.001
    kernel = np.ones((2*side+1, 2*side+1, 2*side+1))
    s = convolve(tomo, kernel, mode="same")
    s2 = convolve(tomosq, kernel, mode="same")
    ns = convolve(ones, kernel, mode="same") + eps

    out = np.sqrt((s2 - s**2 / ns) / ns + eps)
    # out = out>np.std(tomo)*threshold
    out  = out>np.percentile(out, 100-threshold)
    return out.astype(np.uint8)

def boundary_mask(tomo, mask_boundary, binning = 2):
    out = np.zeros(tomo.shape, dtype = np.float32)
    import os
    import sys
    if mask_boundary[-4:] == '.mod':
        os.system('model2point {} {}.point >> /dev/null'.format(mask_boundary, mask_boundary[:-4]))
    else:
        logging.error("mask boundary file should end with .mod but got {} !\n".format(mask_boundary))
        sys.exit()
    
    points = np.loadtxt(mask_boundary[:-4]+'.point', dtype = np.float32)/binning
    
    def get_polygon(points):
        if len(points) == 0:
            logging.info("No polygonal mask")
            return None
        elif len(points) <= 2:
            logging.error("In {}, {} points cannot defines a polygon of mask".format(mask_boundary, len(points)))
            sys.exit()
        else:
            logging.info("In {}, {} points defines a polygon of mask".format(mask_boundary, len(points)))
            return points[:,[1,0]]
    
    if points.ndim < 2: 
        logging.error("In {}, too few points to define a boundary".format(mask_boundary))
        sys.exit()

    z1=points[-2][-1]
    z0=points[-1][-1]

    if abs(z0 - z1) < 5:
        zmin = 0
        zmax = tomo.shape[0]
        polygon = get_polygon(points)
        logging.info("In {}, all points defines a polygon with full range in z".format(mask_boundary))

    else:
        zmin = max(min(z0,z1),0) 
        zmax = min(max(z0,z1),tomo.shape[0])
        polygon = get_polygon(points[:-2])
        logging.info("In {}, the last two points defines the z range of mask".format(mask_boundary))

    '''
    if points.ndim != 2 or points.shape[0] == 1:
        logging.error("In {}, too few points to define a boundary".format(mask_boundary))
        sys.exit()
    elif points.shape[0] == 2 and np.abs(points[-1][-1] - points[-2][-1]) > 5:
        logging.info("In {},the two points defines the z range of mask".format(mask_boundary))
        zmin = max(min(points[-1,-1],points[-2][-1]),0)
        zmax = min(max(points[-1,-1],points[-2][-1]),tomo.shape[0])
        polygon = None
    elif points.shape[0] <5 and np.abs(points[-1][-1] - points[-2][-1]) < 6 :
        logging.error("In {}, {} points can not make a polygon".format(mask_boundary,points.shape[0]-2))
        sys.exit()
    elif points.shape[0] > 4 and np.abs(points[-1][-1] - points[-2][-1]) > 5:
        logging.info("In {}, the last two points defines the z range of mask".format(mask_boundary))
        zmin = max(min(points[-1,-1],points[-2][-1]),0)
        zmax = min(max(points[-1,-1],points[-2][-1]),tomo.shape[0])
        polygon = points[:-2,[1,0]]
    else:
        zmin = 0
        zmax = tomo.shape[0]
        polygon = points[:,[1,0]]
    '''
    zmin = int(zmin)
    zmax = int(zmax)
    if polygon is None:
        out[zmin:zmax,:,:] = 1
    else:
        from matplotlib.path import Path
        poly_path = Path(polygon)
        y, x = np.mgrid[:tomo.shape[1],:tomo.shape[2]]
        coors = np.hstack((y.reshape(-1, 1), x.reshape(-1,1)))
        mask = poly_path.contains_points(coors)
        mask = mask.reshape(tomo.shape[1],tomo.shape[2])
        mask = mask.astype(np.float32)
        out[zmin:zmax,:,:] = mask[np.newaxis,:,:]

    return out

def make_mask(tomo_path, mask_name, mask_boundary = None, side = 5, density_percentage=50., std_percentage=50., surface=None):
    from scipy.ndimage.filters import gaussian_filter
    from skimage.transform import resize
    
    with mrcfile.open(tomo_path, permissive=True) as n:
        header_input = n.header
        #print(header_input)
        pixel_size = n.voxel_size
        tomo = n.data.astype(np.float32)
    sp=np.array(tomo.shape)
    sp2 = sp//2
    bintomo = resize(tomo,sp2,anti_aliasing=True)
  
    gauss = gaussian_filter(bintomo, side/2)
    if density_percentage <=99.8:
        mask1 = maxmask(gauss,side=side, percentile=density_percentage)
    else:
        mask1 = np.ones(sp2)

    if std_percentage <=99.8:
        mask2 = stdmask(gauss,side=side, threshold=std_percentage)
    else:
        mask2 = np.ones(sp2)

    out_mask_bin = np.multiply(mask1,mask2)
   
    if mask_boundary is not None:
        mask3 = boundary_mask(bintomo, mask_boundary)
        out_mask_bin = np.multiply(out_mask_bin, mask3)

    if (surface is not None) and surface < 1:
        for i in range(int(surface*sp2[0])):
            out_mask_bin[i] = 0
        for i in range(int((1-surface)*sp2[0]),sp2[0]):
            out_mask_bin[i] = 0


    out_mask = np.zeros(sp)
    out_mask[0:-1:2,0:-1:2,0:-1:2] = out_mask_bin
    out_mask[0:-1:2,0:-1:2,1::2] = out_mask_bin
    out_mask[0:-1:2,1::2,0:-1:2] = out_mask_bin
    out_mask[0:-1:2,1::2,1::2] = out_mask_bin
    out_mask[1::2,0:-1:2,0:-1:2] = out_mask_bin
    out_mask[1::2,0:-1:2,1::2] = out_mask_bin
    out_mask[1::2,1::2,0:-1:2] = out_mask_bin
    out_mask[1::2,1::2,1::2] = out_mask_bin
    out_mask = (out_mask>0.5).astype(np.uint8)


    with mrcfile.new(mask_name,overwrite=True) as n:
        n.set_data(out_mask)

        n.header.extra2 = header_input.extra2
        n.header.origin = header_input.origin
        n.header.nversion = header_input.nversion
        n.voxel_size = pixel_size
        #print(n.header)
    # with mrcfile.new('./test_mask1.rec',overwrite=True) as n:
    #     n.set_data(mask1.astype(np.float32))    
    # with mrcfile.new('./test_mask2.rec',overwrite=True) as n:
    #     n.set_data(mask2.astype(np.float32))
        n.header.extra2 = header_input.extra2
        n.header.origin = header_input.origin
        n.header.nversion = header_input.nversion
        n.voxel_size = pixel_size
        #print(n.header)
    # with mrcfile.new('./test_mask1.rec',overwrite=True) as n:
    #     n.set_data(mask1.astype(np.float32))    
    # with mrcfile.new('./test_mask2.rec',overwrite=True) as n:
    #     n.set_data(mask2.astype(np.float32))