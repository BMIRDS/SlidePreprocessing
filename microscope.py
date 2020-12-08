"""

What is magnification in wSI:

TL;DR: a resolution of .25mpp are referred to as 40X by convention in digital pathology.
The 40X is technically a magnification power of the objective (objective-power)
with imaginary 10X eyepiece but we just call it "magnification".

"Whole slide images (WSI) are large. A typical sample may be 20mm x 15mm in size, 
and may be digitized with a resolution of .25 micrometers/pixel (conventionally 
described as microns per pixel, or mpp) Most optical microscopes have an eyepiece 
which provides 10X magnification, so using a 40X objective lens actually results 
in 400X magnification. Although instruments which digitize microscope slides do 
not use an eyepiece and may not use microscope objective lenses, by convention 
images captured with a resolution of .25mpp are referred to as 40X, images captured 
with a resolution of .5mpp are referred to as 20X, etc. The resulting image is 
therefore about 80,000 x 60,000 pixels, or 4.8Gp. Images are usually captured with 
24-bit color, so the image data size is about 15GB.""

http://dicom.nema.org/Dicom/DICOMWSI/

"""


def compute_magnification(slide, target_magnification=10):
    """
    Args:
        slide: openslide.OpenSlide object
        target_magnification: Target magnification
            40 for a resolution of .25mpp
            20 for a resolution of .5mpp
            ...

    Return: 
        A dictionary.
        keys:
            'original_magnification': int
            'target_level': int
            'donwsampling_factor': float

    """
    ERR_RANGE = 10  # in %
    ERR_RANGE = ERR_RANGE / 100
    objective_power = None
    if 'openslide.objective-power' in slide.properties:
        objective_power = slide.properties.get('openslide.objective-power')

    elif 'openslide.mpp-x' in slide.properties:
        objective_power = 10 / float(slide.properties.get('openslide.mpp-x'))

    else:
        print("Magnification is not available.")
        # SHOULD THROW ERROR HERE

    """Objective Power is often not accurate; it should typically be 10, 20, or
    40. Thus need rounding. ex) 41.136... -> 40
    """
    objective_power_candidates = [10, 20, 40]
    calibrated = False
    for candidate in objective_power_candidates:
        if candidate * (1 - ERR_RANGE) <= objective_power and\
         objective_power <= candidate * (1 + ERR_RANGE):
            objective_power = candidate
            calibrated = True
            break
    if not calibrated:
        print(f"Magnification is not standard value: {objective_power}")
        # SHOULD THROW ERROR HERE

    """Now compute a donwsampling factor that should be applied to the slide
    to achieve target magnification.

    To reduce the computational cost in later stage, it identifies a downsampling
    level that is already stored in a slide and closest to the target magnification.
    ex)
        Given 40X magnification (0.25 mpp) slide and 5X target magnification,
        it's much easier to downsample 10X image at level1 with a downsampling factor
        of 2 (i.e., 10X / 2 = 5X), rather than downsampling 40X image at level0 with a downsampling
        factor of 8 (i.e., 40X / 8 = 5X) to achieve 5X magnification.

    """
    closest_level = 0
    for l in range(slide.level_count):
        current_magnification = objective_power / int(slide.level_downsamples[l])
        if current_magnification < target_magnification * (1 + ERR_RANGE):
            break
        closest_level = l
    donwsampling_factor = objective_power / int(slide.level_downsamples[closest_level]) / target_magnification
    if donwsampling_factor < 1:
        print("Target magnification is higher than the scanned magnification.")
        # SHOULD THROW ERROR HERE

    results = {
        'original_magnification': objective_power,
        'target_level': closest_level,
        'donwsampling_factor': donwsampling_factor,
    }
    return results


if __name__ == '__main__':
    # test
    from openslide import OpenSlide
    s = OpenSlide('0_testdata/Tumor_002.tif')
    results = compute_magnification(s, 10)
    print(results)








