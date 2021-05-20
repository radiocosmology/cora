

/* This routine does *no* bounds checking. Be careful. */
void _bin_delta_c(
    double* rho,
    int* pixel_ind,
    double* pixel_weight,
    int* radial_ind,
    double* radial_weight,
    double* out,
    int npart,
    int npix,
    int nrad
) {

    #pragma omp parallel for 
    for (int ipart = 0; ipart < npart; ipart++) {

        double v = rho[ipart];

        for (int ipix = 0; ipix < npix; ipix++) {

            int ip = ipart * npix + ipix;

            int pi = pixel_ind[ip];
            double pw = pixel_weight[ip];

            for (int irad  = 0; irad < nrad; irad++) {

                int ir = ipart * nrad + irad;

                int ri = radial_ind[ir];
                double rw = radial_weight[ir];

                if (rw < 0) continue;

                #pragma omp atomic
                out[ri * npix + pi] += v * pw * rw;
            }
        }
    }
}

