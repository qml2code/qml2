void fetch_kernel_move_pointers(double *kernel_element, const double **q1,
                                const double **q2, double half_inv_sigma2,
                                const long long nfeatures) {
  fetch_square_distance_move_pointers(kernel_element, q1, q2, nfeatures);
  *kernel_element = exp(-half_inv_sigma2 * *kernel_element);
}
