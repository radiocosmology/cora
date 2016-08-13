# Change Log

All notable changes to this project will be documented in this file. This
project adheres to [Semantic Versioning](http://semver.org/), with the exception
that I'm using PEP440 to denote pre-releases.

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
