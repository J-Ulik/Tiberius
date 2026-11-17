from astropy.io import fits
import numpy as np
import argparse
from global_utils import parseInput
from scipy.signal import medfilt
from astropy.stats import median_absolute_deviation
from reduction_utils import cosmic_removal as c

parser = argparse.ArgumentParser(description="remove NaN values created after the ramp fit stage")
parser.add_argument('fits_file',help='the base name of the fits file, before any _stage.fits')
#parser.add_argument('--pixel_mask',help='the name of the bad pixel mask file to load in')
args = parser.parse_args()

n1 = fits.open(args.fits_file + "_1_rampfitstep.fits")
n2 = fits.open(args.fits_file + "_1overfstep.fits")

print("NaN contamination:", round( np.sum(np.isnan(n1["SCI"].data[0]))/(n1["SCI"].shape[-1]*n1["SCI"].shape[-2]*0.01) ,2) ,"%")
nan_mask = np.isnan(n1["SCI"].data[0])

# Load in reference files
home = "/data2/ju126/data"
instrument = n1[0].header["INSTRUME"].lower()

try:
    dark_name = n1[0].header["R_DARK"].split("//")[1]
except:
    dark_name = n1[0].header["R_DARK"]
dark_file = fits.open("%s/crds_cache/jwst/references/jwst/%s/%s"%(home,instrument,dark_name))
dark = dark_file["DRK"].data

try:
    gain_name = n1[0].header["R_GAIN"].split("//")[1]
except:
    gain_name = n1[0].header["R_GAIN"]
gain_file = fits.open("%s/crds_cache/jwst/references/jwst/%s/%s"%(home,instrument,gain_name))
gain = gain_file["SCI"].data[gain_file[0].header["SUBSTRT1"]:gain_file[0].header["SUBSTRT1"]+dark.shape[0], gain_file[0].header["SUBSTRT2"]:gain_file[0].header["SUBSTRT2"]+dark.shape[1]]

# Looking at headers, read noise already looks to take into accout difference between frames?
try:
    readnoise_name = n1[0].header["R_READNO"].split("//")[1]
except:
    readnoise_name = n1[0].header["R_READNO"]
readnoise_file = fits.open("%s/crds_cache/jwst/references/jwst/%s/%s"%(home,instrument,readnoise_name))
readnoise = readnoise_file["SCI"].data[readnoise_file[0].header["SUBSTRT1"]:readnoise_file[0].header["SUBSTRT1"]+dark.shape[0], readnoise_file[0].header["SUBSTRT2"]:readnoise_file[0].header["SUBSTRT2"]+dark.shape[1]]


# Slope_est calculation for use in error calculations
slope_calc = n2["SCI"].data.astype(np.float32)
unsaturated = n2["GROUPDQ"].data & 2 == 0
slope_calc[~unsaturated] = np.nan
differences = abs(np.diff(slope_calc, axis=1))
slope_est = np.nanmedian(differences, axis=(0, 1))
tgroup = n1[0].header["TGROUP"]

# Manual ramp fitting of NaN values
for integration in range(n1["SCI"].data.shape[0]):
    times = (60*60*24)*np.linspace(0, n2["INT_TIMES"].data[integration][3]-n2["INT_TIMES"].data[integration][1], num=n2["SCI"].data.shape[1])

    for (x,y),val in np.ndenumerate(nan_mask):
        if val==True:
            # Saturation check
            unsaturated = n2["GROUPDQ"].data[integration,:,x,y] & 2 == 0
            ngroups = max(2, len(unsaturated))

            # https://stcal.readthedocs.io/en/latest/stcal/ramp_fitting/description.html#ramp-output-products
            # No segment weighting is required as will only have one segment after removing saturated pixels (which are always at the end)
            n1["SCI"].data[integration, x,y] = np.polyfit(times[unsaturated] ,n2["SCI"].data[integration,:,x,y][unsaturated], 1)[0].astype(np.float32)

            # Errors!
            rnoise = (12*readnoise[x,y]**2 / (ngroups**3-ngroups)*(tgroup**2)*(gain[x,y]**2) )
            poisson = (slope_est[x,y]+dark[x,y])/(tgroup*gain[x,y]*(ngroups-1))
            n1["VAR_RNOISE"].data[integration, x,y] = rnoise
            n1["VAR_POISSON"].data[integration, x,y] = poisson
            n1["ERR"].data[integration, x,y] = np.sqrt(poisson + rnoise)


# Replacing values which are wildly different with neigbouring values
mf_box_width = 7
median_frame = np.median(n1["SCI"].data, axis=0)
median = np.array([medfilt(row,mf_box_width) for row in median_frame])
residuals = median - median_frame

n1_mask = c.flag_bad_pixels(median_frame,verbose=False,std_box_width=51,mf_box_width=mf_box_width,cut_off=5,use_mad=False,max_pixels_per_row = 200)

for integration in range(n1["SCI"].data.shape[0]):
    for (x,y),val in np.ndenumerate(n1_mask):
        if val==True:
            if y==0:
                n1["SCI"].data[integration, x,y] = n1["SCI"].data[integration, x,y+1]
                n1["VAR_RNOISE"].data[integration, x,y] = n1["VAR_RNOISE"].data[integration, x,y+1]
                n1["VAR_POISSON"].data[integration, x,y] = n1["VAR_POISSON"].data[integration, x,y+1]
            elif y==n1["SCI"].data.shape[2]:
                n1["SCI"].data[integration, x,y] = n1["SCI"].data[integration, x,y-1]
                n1["VAR_RNOISE"].data[integration, x,y] = n1["VAR_RNOISE"].data[integration, x,y-1]
                n1["VAR_POISSON"].data[integration, x,y] = n1["VAR_POISSON"].data[integration, x,y-1]
            else:
                n1["SCI"].data[integration, x,y] = (n1["SCI"].data[integration, x,y-1] + n1["SCI"].data[integration, x,y+1])/2
                n1["VAR_RNOISE"].data[integration, x,y] = (n1["VAR_RNOISE"].data[integration, x,y-1] + n1["VAR_RNOISE"].data[integration, x,y+1])/2
                n1["VAR_POISSON"].data[integration, x,y] = (n1["VAR_POISSON"].data[integration, x,y-1] + n1["VAR_POISSON"].data[integration, x,y+1])/2

            n1["ERR"].data[integration,x,y] = np.sqrt(n1["VAR_RNOISE"].data[integration, x,y]+n1["VAR_POISSON"].data[integration, x,y])

new_name = args.fits_file + "_nancleaningstep.fits"
n1.writeto(new_name,overwrite=True)
