# [20.5.0](https://github.com/radiocosmology/cora/compare/v3.0.0...v20.5.0) (2020-05-05)


### Bug Fixes

* typo and remove unnecessary dependency on caput ([57e9c12](https://github.com/radiocosmology/cora/commit/57e9c1285cc0c0ead14dc4b1202da5c8c7535e42))
* workaround a healpy import bug, and fix for a h5py warning ([e8d1666](https://github.com/radiocosmology/cora/commit/e8d1666150c906a293ca9d98118785352532eb7a))
* **corr21cm:** error in T_b for EoR 21 cm signal. ([15683c3](https://github.com/radiocosmology/cora/commit/15683c3de26b0e53ac6f8b8eb3f784825042bd04))
* **makesky:** enforce the galactic maps must be generated with more than one freq ([e3da161](https://github.com/radiocosmology/cora/commit/e3da1618c50fc41f8e7ac14ca38b1290fac66af2))
* **makesky:** work around h5py string attribute handling ([d8cdbbc](https://github.com/radiocosmology/cora/commit/d8cdbbc2e59845295f727dc94f65aca31744d0a8))
* **makesky:** print meaningful message when file already exists ([c252b84](https://github.com/radiocosmology/cora/commit/c252b84626dadde5c49458258cfb9284a71a40f8))


### Features

* **makesky:** change binning to order used in driftscan ([17ad8a9](https://github.com/radiocosmology/cora/commit/17ad8a94b4dd17b9f19e0460e1bc11c5d64c5b7d))
* **makesky**: add ability to select a specific list of channels ([781a13c](https://github.com/radiocosmology/cora/commit/781a13c9c437b0ad6991611c01f06e06a6209c1a))
* **singlesource:** add a command to generate a map containing a single source ([b6d7934](https://github.com/radiocosmology/cora/commit/b6d793412ffeab70a6e24ac03b0b92cb682ae5db))
* **versioneer:** add versioneer for improved version naming ([596b45d](https://github.com/radiocosmology/cora/commit/596b45de44ae55ddc8febfd27c028eb417f16bd9))



## [3.0.0] - 2019-01-07

### Changed

- Python 3 support. This may break some things, please report any issues you
  have.
- Setup travis integration.


## [2.1.2] - 2019-01-07

### Added

- Basic unit tests checking the calculation of correlation functions and map
  generation.

### Fixed

- Major bug in drawing a realisation of a power spectrum. This my have
  significantly impacted simulated maps since 2.1.1.


## [2.1.1] - 2017-08-02

### Added

- Added the ability to control the oversampling done when generating
  realisations of multi-frequency angular power spectra.
- A new module for fast, parallel, bi-linear interpolation.
- The ability to choose the spectral index map when simulating the
  galaxy.

### Changed

- Significant optimisations to the calculation of cosmological angular power spectra
- Speed ups in cosmological distance calculation.
- PEP8 fixes
- Refactored to protect GSL import if it wasn't installed


## [2.1.0] - 2016-11-24

### Added

- Routines for calculating spherical harmonics accurately.

### Changed

- The routines in `cora.util.sphbessel` have been moved into `cora.util.sphfunc`.
- There is now an optional dependency on `pygsl`.
- Some old files have been moved from the repository.


## [2.0.0] - 2016-08-13

### Added

- This `CHANGELOG` file.

### Changed

- Significant changes to the generation of galactic polarisation. Now has a much
  more robust choice of Faraday space grid, at the risk of being more memory
  intensive.
- Improved the output format to match that of `draco.containers.Map`.
- Large changes to the `cora-makesky` script to improve flexibility for choosing
  frequencies. This has lead to non backwards compatible changes to the command
  line options.
- Made the version sourced from a single location to fix the inconsistencies.
- The license to MIT.
