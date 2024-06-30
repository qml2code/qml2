void skip_molecules_wcharge(const long long num_skipped_mols,
                            const long long nfeatures,
                            const long long **cur_natoms_pointer,
                            const double **feature_pointer,
                            const long long **cur_ncharges_pointer,
                            const bool extra_natoms_move) {
  // Skip molecules current batch is not supposed to work with
  if (extra_natoms_move) {
    (*cur_natoms_pointer)++;
  }
  if (num_skipped_mols == 0) {
    return;
  }
  long long atom_shift = 0;
  for (long long i = 0; i != num_skipped_mols; i++) {
    atom_shift += **cur_natoms_pointer;
    (*cur_natoms_pointer)++;
  }
  (*feature_pointer) += atom_shift * nfeatures;
  (*cur_ncharges_pointer) += atom_shift;
}

double square(const double val) { return val * val; }

void set_kernel_to_zero(double *kernel_matrix, long long start_mol1,
                        long long start_mol2, long long T1, long long T2,
                        long long nm1, long long nm2,
                        long long kernel_output_start_pointer_mol1_step) {
  // Initially the kernel value is 0.
  double *kernel_start_pointer = &kernel_matrix[start_mol1 * nm2 + start_mol2];
  double *kernel_pointer;
  for (long long mol1 = start_mol1; mol1 < nm1; mol1 += T1) {
    kernel_pointer = kernel_start_pointer;
    for (long long mol2 = start_mol2; mol2 < nm2; mol2 += T2) {
      *kernel_pointer = 0.0;
      kernel_pointer += T2;
    }
    kernel_start_pointer += kernel_output_start_pointer_mol1_step;
  }
}

// Determine the starting ID and the stride over molecule counters based on
// block and threads parameters. If the number of threads among one direction
// exceeds the number of molecules the extra threads are cut more or less evenly
// amond different streaming multiprocessors.
void set_total_thread_numbers_ids_1D(long long nthreads, long long thread_id,
                                     long long nblocks, long long block_id,
                                     long long *tot_thread_id,
                                     long long *tot_num_threads) {
  *tot_thread_id = thread_id * nblocks + block_id;
  *tot_num_threads = nblocks * nthreads;
}

void set_total_thread_numbers_ids(long long *thread_id1, long long *thread_id2,
                                  long long *num_threads1,
                                  long long *num_threads2) {
  set_total_thread_numbers_ids_1D(blockDim.x, threadIdx.x, gridDim.x,
                                  blockIdx.x, thread_id1, num_threads1);
  set_total_thread_numbers_ids_1D(blockDim.y, threadIdx.y, gridDim.y,
                                  blockIdx.y, thread_id2, num_threads2);
}

void add_gaussian_move_pointers(double *kernel_element, double *sqdist,
                                const double **q1, const double **q2,
                                double inv_sigma2, const long long nfeatures) {
  *sqdist = 0.0;
  for (long long feature_id = 0; feature_id != nfeatures; ++feature_id) {
    *sqdist += square(**q1 - **q2);
    (*q1)++;
    (*q2)++;
  }
  *kernel_element += exp(inv_sigma2 * *sqdist);
}

void skip_molecules(const long long num_skipped_mols, const long long nfeatures,
                    const long long **cur_natoms_pointer,
                    const double **feature_pointer,
                    const bool extra_natoms_move) {
  if (extra_natoms_move) {
    (*cur_natoms_pointer)++;
  }
  if (num_skipped_mols == 0) {
    return;
  }
  long long atom_shift = 0;
  for (long long i = 0; i != num_skipped_mols; i++) {
    atom_shift += **cur_natoms_pointer;
    (*cur_natoms_pointer)++;
  }
  (*feature_pointer) += atom_shift * nfeatures;
}

extern "C" __global__
    // K.Karand: 1.TBH I started optimizing memory access and then realized I
    // was having a lot of fun with C pointers. The resulting complexity may be
    // overkill.
    // 2. The code's parallelization does not distinguish between blocks and
    // threads, following the example here:
    //  https://developer.nvidia.com/blog/even-easier-introduction-cuda/
    // I think the parallelization can be imporved upon, but it's good enough
    // for GPU power demonstration.
    // 3. I use long long instead of int almost everywhere because np.array by
    // default uses np.int64 which, if I understand correctly, corresponds to
    // long long in C. Might be overcautious.
    void
    local_dn_gaussian_kernel(const double *q1, const double *q2,
                             const long long *n1, const long long *n2,
                             const long long *ncharges1,
                             const long long *ncharges2, const double sigma,
                             const long long nm1, const long long nm2,
                             const long long nfeatures, double *kernel_matrix) {
  // K.Karand: I see two ways to divide the q1 molecules: a) divide q1 into
  // unbrocken intervals b) iterate over q1 molecules with a step dependent on
  // number of threads. a) squeezes a more speed (not sure how much) if the
  // input molecule list is unordered, b) does not break down badly if the list
  // is ordered, say, from smallest to largest molecules. I went for b) for
  // user-friendliness, but not %100 sure it's the best option.

  // Distribute starting molecule ids according to the threads.
  long long mol1, starting_mol2, T1, T2;
  set_total_thread_numbers_ids(&mol1, &starting_mol2, &T1, &T2);
  long long mol2 = starting_mol2;
  // If some threads are ``extra'' terminate them.
  if (mol1 >= nm1 || mol2 >= nm2) {
    return;
  }
  // Thread numbers for parallelization over q1 and q2.

  long long last_nm1_ubound = nm1 - T1;
  long long last_nm2_ubound = nm2 - T2;

  // kernel_output_pointer is where atom-atom covariance is added as q2 is
  // cycled through. kernel_output_start_pointer is where kernel_output_pointer
  // is reset at completion of loop over q2.
  double *kernel_output_start_pointer = &kernel_matrix[mol1 * nm2 + mol2];
  double *kernel_output_pointer = kernel_output_start_pointer;
  long long kernel_output_start_pointer_mol1_step = nm2 * T1;

  // Pointers towards relevant input, moved as mol1 and mol2 change.
  const long long *cur_n1_pointer = &n1[0];
  const double *cur_q1_pointer = &q1[0];
  const long long *cur_ncharges1_pointer = &ncharges1[0];
  skip_molecules_wcharge(mol1, nfeatures, &cur_n1_pointer, &cur_q1_pointer,
                         &cur_ncharges1_pointer, false);
  // starting pointers where mols 2 are reset after cycling through them is done
  const long long *n2_start_pointer = &n2[0];
  const double *q2_start_pointer = &q2[0];
  const long long *ncharges2_start_pointer = &ncharges2[0];
  skip_molecules_wcharge(starting_mol2, nfeatures, &n2_start_pointer,
                         &q2_start_pointer, &ncharges2_start_pointer, false);
  const long long *cur_n2_pointer = n2_start_pointer;
  const double *cur_q2_pointer = q2_start_pointer;
  const long long *cur_ncharges2_pointer = ncharges2_start_pointer;

  // Atom counters.
  long long atom1 = 0; // counter for atom considered in mol1
  long long atom2 = 0; // counter for atom considered in mol2

  double sqdist = 0.0; // where currently calculated square dist is stored.
  double inv_sigma2 = -0.5 / square(sigma); // sqdist's multiplier in Gaussian.

  // Set all considered kernel elements to 0.0.
  set_kernel_to_zero(kernel_matrix, mol1, mol2, T1, T2, nm1, nm2,
                     kernel_output_start_pointer_mol1_step);

  while (true) {
    if (*cur_ncharges1_pointer == *cur_ncharges2_pointer) {
      add_gaussian_move_pointers(kernel_output_pointer, &sqdist,
                                 &cur_q1_pointer, &cur_q2_pointer, inv_sigma2,
                                 nfeatures);
    } else {
      cur_q1_pointer += nfeatures;
      cur_q2_pointer += nfeatures;
    };
    atom2++; // proceed to calculate next atom
    cur_ncharges2_pointer++;
    if (atom2 >= *cur_n2_pointer) { // move to next molecule
      atom2 = 0;
      if (mol2 >= last_nm2_ubound) { // move to next atom in molecule 1 and
                                     // first molecule 2.
        mol2 = starting_mol2;
        cur_n2_pointer = n2_start_pointer;
        cur_q2_pointer = q2_start_pointer;
        cur_ncharges2_pointer = ncharges2_start_pointer;
        atom1++;
        cur_ncharges1_pointer++;
        if (atom1 >= *cur_n1_pointer) {
          // move to next molecule 1
          atom1 = 0;
          if (mol1 >= last_nm1_ubound) { // calculation finished
            return;
          }
          skip_molecules_wcharge(T1 - 1, nfeatures, &cur_n1_pointer,
                                 &cur_q1_pointer, &cur_ncharges1_pointer, true);
          mol1 += T1;
          kernel_output_start_pointer += kernel_output_start_pointer_mol1_step;
        };
        // resetting the kernel pointer
        kernel_output_pointer = kernel_output_start_pointer;
        continue;
      } else {
        // move to next molecule 2.
        skip_molecules_wcharge(T2 - 1, nfeatures, &cur_n2_pointer,
                               &cur_q2_pointer, &cur_ncharges2_pointer, true);
        mol2 += T2;
        kernel_output_pointer += T2;
      }
    }
    cur_q1_pointer -= nfeatures; // not sure whether this is faster than saving
                                 // a pointer and re-pointing to it
  }
}

extern "C" __global__ void
local_gaussian_kernel(const double *q1, const double *q2, const long long *n1,
                      const long long *n2, const double sigma,
                      const long long nm1, const long long nm2,
                      const long long nfeatures, double *kernel_matrix) {
  // K.Karand: I see two ways to divide the q1 molecules: a) divide q1 into
  // unbrocken intervals b) iterate over q1 molecules with a step dependent on
  // number of threads. a) squeezes a more speed (not sure how much) if the
  // input molecule list is unordered, b) does not break down badly if the list
  // is ordered, say, from smallest to largest molecules. I went for b) for
  // user-friendliness, but not %100 sure it's the best option.

  // Distribute starting molecule ids according to the threads.
  long long mol1, starting_mol2, T1, T2;
  set_total_thread_numbers_ids(&mol1, &starting_mol2, &T1, &T2);
  long long mol2 = starting_mol2;
  // If some threads are ``extra'' terminate them.
  if (mol1 >= nm1 || mol2 >= nm2) {
    return;
  }
  // Thread numbers for parallelization over q1 and q2.

  long long last_nm1_ubound = nm1 - T1;
  long long last_nm2_ubound = nm2 - T2;

  // kernel_output_pointer is where atom-atom covariance is added as q2 is
  // cycled through. kernel_output_start_pointer is where kernel_output_pointer
  // is reset at completion of loop over q2.
  double *kernel_output_start_pointer = &kernel_matrix[mol1 * nm2 + mol2];
  double *kernel_output_pointer = kernel_output_start_pointer;
  long long kernel_output_start_pointer_mol1_step = nm2 * T1;

  // Pointers towards relevant input, moved as mol1 and mol2 change.
  // starting pointers where mols 2 are reset after cycling through them is done
  const long long *cur_n1_pointer = &n1[0];
  const double *cur_q1_pointer = &q1[0];
  skip_molecules(mol1, nfeatures, &cur_n1_pointer, &cur_q1_pointer, false);

  const long long *n2_start_pointer = &n2[0];
  const double *q2_start_pointer = &q2[0];
  skip_molecules(starting_mol2, nfeatures, &n2_start_pointer, &q2_start_pointer,
                 false);
  const long long *cur_n2_pointer = n2_start_pointer;
  const double *cur_q2_pointer = q2_start_pointer;

  // Atom counters.
  long long atom1 = 0; // counter for atom considered in mol1
  long long atom2 = 0; // counter for atom considered in mol2

  double sqdist = 0.0; // where currently calculated square dist is stored.
  double inv_sigma2 = -0.5 / square(sigma); // sqdist's multiplier in Gaussian.

  // Set all considered kernel elements to 0.0.
  set_kernel_to_zero(kernel_matrix, mol1, mol2, T1, T2, nm1, nm2,
                     kernel_output_start_pointer_mol1_step);

  while (true) {
    add_gaussian_move_pointers(kernel_output_pointer, &sqdist, &cur_q1_pointer,
                               &cur_q2_pointer, inv_sigma2, nfeatures);
    atom2++;                        // proceed to calculate next atom
    if (atom2 >= *cur_n2_pointer) { // move to next molecule
      atom2 = 0;
      if (mol2 >= last_nm2_ubound) { // move to next atom in molecule 1 and
                                     // first molecule 2.
        mol2 = starting_mol2;
        cur_n2_pointer = n2_start_pointer;
        cur_q2_pointer = q2_start_pointer;
        atom1++;
        if (atom1 >= *cur_n1_pointer) {
          // move to next molecule 1
          atom1 = 0;
          if (mol1 >= last_nm1_ubound) { // calculation finished
            return;
          }
          skip_molecules(T1 - 1, nfeatures, &cur_n1_pointer, &cur_q1_pointer,
                         true);
          mol1 += T1;
          kernel_output_start_pointer += kernel_output_start_pointer_mol1_step;
        };
        // resetting the kernel pointer
        kernel_output_pointer = kernel_output_start_pointer;
        continue;
      } else {
        // move to next molecule 2.
        skip_molecules(T2 - 1, nfeatures, &cur_n2_pointer, &cur_q2_pointer,
                       true);
        mol2 += T2;
        kernel_output_pointer += T2;
      }
    }
    cur_q1_pointer -= nfeatures; // not sure whether this is faster than saving
                                 // a pointer and re-pointing to it
  }
}

extern "C" __global__
    // KK: Largely copy-pasted from local_dn_gaussian_kernel; if proven relevant
    // should be rewritten.
    void
    gaussian_kernel(const double *q1, const double *q2, const double sigma,
                    const long long nm1, const long long nm2,
                    const long long nfeatures, double *kernel_matrix) {

  // Distribute starting molecule ids according to the threads.
  long long mol1, starting_mol2, T1, T2;
  set_total_thread_numbers_ids(&mol1, &starting_mol2, &T1, &T2);
  long long mol2 = starting_mol2;
  // If some threads are ``extra'' terminate them.
  if (mol1 >= nm1 || mol2 >= nm2) {
    return;
  }
  // Thread numbers for parallelization over q1 and q2.

  long long last_nm1_ubound = nm1 - T1;
  long long last_nm2_ubound = nm2 - T2;

  // kernel_output_pointer is where atom-atom covariance is added as q2 is
  // cycled through. kernel_output_start_pointer is where kernel_output_pointer
  // is reset at completion of loop over q2.
  double *kernel_output_pointer = &kernel_matrix[mol1 * nm2 + mol2];
  long long kernel_output_mol1_step = nm2 * (T1 - 1) + T2;
  long long q2_mol_step = (T2 - 1) * nfeatures;
  long long q1_mol_step = (T1 - 1) * nfeatures;
  // Pointers towards relevant input, moved as mol1 and mol2 change.
  const double *cur_q1_pointer = &q1[mol1 * nfeatures];
  // starting pointers where mols 2 are reset after cycling through them is done
  const double *q2_start_pointer = &q2[mol2 * nfeatures];
  const double *cur_q2_pointer = q2_start_pointer;

  double sqdist = 0.0; // where currently calculated square dist is stored.
  double inv_sigma2 = -0.5 / square(sigma); // sqdist's multiplier in Gaussian.
  while (true) {
    *kernel_output_pointer = 0.0;
    add_gaussian_move_pointers(kernel_output_pointer, &sqdist, &cur_q1_pointer,
                               &cur_q2_pointer, inv_sigma2, nfeatures);
    if (mol2 >= last_nm2_ubound) {
      mol2 = starting_mol2;
      cur_q2_pointer = q2_start_pointer;
      if (mol1 >= last_nm1_ubound) { // calculation finished
        return;
      }
      mol1 += T1;
      kernel_output_pointer += kernel_output_mol1_step;
      cur_q1_pointer += q1_mol_step;
    } else {
      mol2 += T2;
      kernel_output_pointer += T2;
      cur_q2_pointer += q2_mol_step;
      cur_q1_pointer -= nfeatures;
    };
  };
  cur_q1_pointer -= nfeatures; // not sure whether this is faster than saving
                               // a pointer and re-pointing to it
}
