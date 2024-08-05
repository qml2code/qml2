void fetch_kernel_move_pointers(double *kernel_element, const double **q1,
                                const double **q2, double inv_sigma,
                                const long long nfeatures) {
  fetch_distance_move_pointers(kernel_element, q1, q2, nfeatures);
  *kernel_element = exp(-inv_sigma * *kernel_element);
}
