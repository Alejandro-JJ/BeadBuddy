from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import pyclesperanto_prototype as cle
import numpy as np


def MasterSegmenter(img_path, timepoint=0, backg_r=20, threshold=200, spot_sigma=1, outline_sigma=1, 
                    perc_int=100, show_plots=False, savepics=False):
    """
    Master Segmenter Function
    
    Many curated steps that render nice segmentations of PAA Beads inside Zebrafish
    embryos. The user has 5 main parameters to tune,explained below:
    After innumerable trials and errors, this seems like a sound code.
    All steps are absolutely necessary to achieve a proper segmentation.
    
    img_path = str, absolute path of .tif with a single channel, isotropic resolution
    back_r = float, box size for background cleaning, 10-20 is good
    threshold = HARD threshold for the whole image, probably the most critical parameter
    spot_sigma = approx size of searched blobs
    outline_sigma = tightness of segm. 1 is the desired value, any higher increases virtually the volume
    perc_int = how much percentage of the brightest segmentations we want to retain
    show_plots = self explanatory
      
    """   
    # Make sure we have a GPU, simply the first one to pop up
    #cle.select_device(cle.available_device_names()[0])
    
    # Read and mount the image on the GPU
    im = imread(img_path)
    input_gpu = cle.push(im)
    
    # Clean background
    im_background_clean = cle.top_hat_box(input_gpu, radius_x=backg_r, 
                                          radius_y=backg_r, radius_z=backg_r)
    
    # Mask with a hard threshold
    mask = cle.threshold(im_background_clean, constant=threshold)
    im_masked = im_background_clean*mask
    
    # Segment with PyCLesperanto method
    im_segmented = cle.voronoi_otsu_labeling(im_masked, spot_sigma=spot_sigma, 
                                             outline_sigma=outline_sigma)

    # Extract a certain top percentage of intensities, in case we have speckles
    intensity_map = cle.mean_intensity_map(im_background_clean, im_segmented)
    min_intensity = (100-perc_int)/100*np.amax(intensity_map)
    
    # Final product
    im_beads = cle.exclude_labels_with_map_values_out_of_range(intensity_map, im_segmented, 
                                                        minimum_value_range=min_intensity, 
                                                        maximum_value_range=np.inf)
    im_beads = cle.pull(im_beads)
    
    # Some beads statistics
    # Size takes for granted 1x1x1 um resolution, should be adapted if not
    (bead, size) = np.unique(im_beads, return_counts=True)
    bead = bead[1:]
    size = size[1:] # delete contribution from background pixel value=0
    n = len(bead)
    radii = np.cbrt(3*size/(4*np.pi))
    #meansize = np.mean(size) if len(size)>0 else 0 # in case we dont have any bead
    #meanradius = np.cbrt(3*meansize/(4*np.pi))

    # Optinal plots of the several steps (useful to find the threshold value)
    if show_plots==True:
        fig, ax = plt.subplots(1,3,sharey=True,figsize=(12,6))
        plt.tight_layout()#, plt.suptitle(name, fontsize=14)
        cle.imshow(im_background_clean, plot=ax[0]), ax[0].title.set_text('Background corrected')
        cle.imshow(im_masked, plot=ax[1]), ax[1].title.set_text('Masked')
        cle.imshow(im_beads, plot=ax[2], labels=True), ax[2].title.set_text('Final segmentation')
        plt.show()
        
    # Optional save pictures
    if savepics==True:
        savename = 'BinaryBeads_tp_'+str(timepoint).zfill(2)+'.tif'
        imsave(savename, im_beads.astype(bool)) # save binarized version 
    
    # do I need to pull??
    return im_beads, n, radii