__constant__ double sqrt5 = 2.23606797749979; // sqrt(5.);

void fetch_kernel_move_pointers(double *kernel_element, const double **q1,
                                const double **q2, double inv_sigma,
                                const long long nfeatures) {
  fetch_distance_move_pointers(kernel_element, q1, q2, nfeatures);
  *kernel_element *= inv_sigma;
  *kernel_element = exp(-*kernel_element * sqrt5) *
                    (1 + sqrt5 * *kernel_element +
                     *kernel_element * *kernel_element * 5. / 3.);
}
