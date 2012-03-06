from simulations import corr21cm, ps_estimation



cr = corr21cm.Corr21cm()
#cr.add_mean = True

# Reduce the number of pixels to make faster
cr.x_num = 64
cr.y_num = 64
cr.nu_num = 64

# Adjust angular and frequency ranges to make close to a cube.
cr.x_width = 10.0   # degrees
cr.y_width = 10.0   # degrees
cr.nu_lower = 500.0 # MHz
cr.nu_upper = 600.0 # MHz

# Generate two realisations
f1 = cr.getfield()
f2 = cr.getfield()

# Calculate 2D statistics (that is treating kpar and kperp
# separately).

# 2D powerspectra of each field
ps1 = ps_estimation.ps_azimuth(f1, window = True, kmodes = False)
ps2 = ps_estimation.ps_azimuth(f2, window = True, kmodes = False)
# 2D cross power spectrum
ps12 = ps_estimation.crossps_azimuth(f1, f2, window = True, kmodes = False)

# Rescale to make differences more obvious
ch1 = ps12 / (ps1 * ps2)**0.5

# 2D mutual coherence of the two fields
coh = np.abs(ch1)**2

## Calculate 1D radially average statistics.

## These do not account for anisotropy (but small as angular and
## frequency ranges chosen to be near a cube)

# Get 1D power spectra of each field
pst1, kv = ps_estimation.ps(f1)
pst2, kv = ps_estimation.ps(f2)

# Get 1d cross power spectrum
pst12, kv = ps_estimation.crossps(f1, f2)

# Scaled cross spectrum
cht1 = pst12 / (pst1 * pst2)**0.5

# 1D mutual coherence
coht = np.abs(cht1)**2
