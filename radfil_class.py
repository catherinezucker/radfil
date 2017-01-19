class radfil(object):

    """
    Container object which stores the required metadata for building the radial profiles
    
    Parameters
    ------
    img : numpy.ndarray or astropy.io.fits.PrimaryHDU
        A 2D array of the image data
    mask: numpy.ndarray or astropy.io.fits.PrimaryHDU
        A 2D array of the mask data; must be of boolean type
        and the same shape as img arrray
    distance : float
        Distance to the filament; must be entered in pc
    imgscale : float, optional
        The physical size of each pixel in your image (in pc); only required if
        header is not provided
    skeleton: numpy.ndarray or astropy.io.fits.PrimaryHDU, optional
        A 2D array of the filament spine; must be of boolean type
        and the same shape as img array. If array is not provided,
        the program will create a skeleton using the Filfinder package
        (https://github.com/e-koch/FilFinder.git)
    
        
        
    
