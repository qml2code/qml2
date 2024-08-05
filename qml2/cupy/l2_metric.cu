double square(const double val) { return val * val; }

void fetch_square_distance_move_pointers(double *output, const double **q1,
                                         const double **q2,
                                         const long long nfeatures) {
  *output = 0.0;
  for (long long feature_id = 0; feature_id != nfeatures; ++feature_id) {
    *output += square(**q1 - **q2);
    (*q1)++;
    (*q2)++;
  }
}

void fetch_distance_move_pointers(double *output, const double **q1,
                                  const double **q2,
                                  const long long nfeatures) {
  fetch_square_distance_move_pointers(output, q1, q2, nfeatures);
  *output = sqrt(*output);
}
