import camb

from cora.signal import lsscontainers
from cora.util import cosmology


# Cosmological params taken from Planck 2018: Planck-TT,TE,EE+lowL+lensing+BAO
background_params = {
    "ombh2": 0.02242,
    "omch2": 0.11933,
    "H0": 67.66,
    "omk": 0.0,
    "TCMB": 2.7255,
    "nnu": 3.046,
}

# Common CAMB params
base_params = {
    # Initial power spectrum
    "As": 2.105e-9,
    "ns": 0.9665,
    # CAMB params
    "WantTransfer": True,
    "WantCls": False,
    # Matter PS params
    "redshifts": (1.0,),
    "kmax": 2000.0,
}

extended_params = {
    "planck2018_z1.0_linear": {
        "nonlinear": False,
    },
    "planck2018_z1.0_halofit-original": {
        "nonlinear": True,
        "halofit_version": "original",
    },
    "planck2018_z1.0_halofit-mead": {
        "nonlinear": True,
        "halofit_version": "mead2020",
    },
    "planck2018_z1.0_halofit-mead-feedback": {
        "nonlinear": True,
        "halofit_version": "mead2020_feedback",
    },
    "planck2018_z1.0_halofit-takahashi": {
        "nonlinear": True,
        "halofit_version": "takahashi",
    },
}

c = cosmology.Cosmology.from_physical(**background_params)

for name, p in extended_params.items():
    print(f"Generating {name}")

    full_params = dict(**background_params, **base_params, **p)

    cp = camb.set_params(**full_params)
    r = camb.get_results(cp)

    k, _, pk = r.get_matter_power_spectrum(minkh=1e-4, maxkh=1000, npoints=1024)

    pkc = lsscontainers.MatterPowerSpectrum(
        k, pk, ps_redshift=base_params["redshifts"][0], cosmology=c
    )
    pkc.attrs["camb_params"] = full_params
    pkc.attrs["tag"] = name

    fname = f"ps_{name}.h5"
    print(f"Saving {fname}")
    pkc.save(fname)
