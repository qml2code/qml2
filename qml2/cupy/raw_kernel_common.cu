
void set_total_thread_numbers_ids_1D_local(
    long long nthreads, long long thread_id, long long nblocks,
    long long block_id, const long long *natoms, const long long num_mols,
    long long *start_mol, long long *finish_mol) {

  long long tot_nthreads = nblocks * nthreads;
  long long tot_thread_id = nthreads * block_id + thread_id;

  long long tot_natoms = 0;
  for (long long i = 0; i != num_mols; i++) {
    tot_natoms += natoms[i];
  }
  // How many atoms roughly should be evaluated by a thread
  long long workload = tot_natoms / tot_nthreads;
  long long current_remaining_natoms = workload;
  long long current_thread_id = 0;
  long long cur_natoms;
  *start_mol = 0;
  long long current_mol = 0;
  while (true) {
    if (current_mol == num_mols - 1) {
      if (current_thread_id == tot_thread_id) {
        *finish_mol = current_mol;
      } else {
        *start_mol = -1;
      };
      return;
    };
    cur_natoms = natoms[current_mol];
    current_remaining_natoms -= cur_natoms;
    if (current_remaining_natoms <= 0) {
      if (*start_mol == current_mol) {
        *finish_mol = *start_mol;
      } else {
        if (current_remaining_natoms < -cur_natoms / 2 ||
            current_mol == num_mols - 1) {
          *finish_mol = current_mol - 1;
        } else {
          *finish_mol = current_mol;
        }
      }
      if (current_thread_id == tot_thread_id) {
        return;
      }
      current_thread_id++;
      *start_mol = *finish_mol + 1;
      tot_natoms -= workload + current_remaining_natoms;
      workload = tot_natoms / (tot_nthreads - current_thread_id);
      current_remaining_natoms = workload;
    }
    current_mol++;
  }
}

void set_total_thread_numbers_ids_local(
    long long *start_mol1, long long *start_mol2, long long *finish_mol1,
    long long *finish_mol2, const long long *natoms1, const long long *natoms2,
    const long long nm1, const long long nm2) {
  set_total_thread_numbers_ids_1D_local(blockDim.x, threadIdx.x, gridDim.x,
                                        blockIdx.x, natoms1, nm1, start_mol1,
                                        finish_mol1);
  set_total_thread_numbers_ids_1D_local(blockDim.y, threadIdx.y, gridDim.y,
                                        blockIdx.y, natoms2, nm2, start_mol2,
                                        finish_mol2);
}

void skip_molecules_wcharge(const long long num_skipped_mols,
                            const long long nfeatures,
                            const long long **cur_natoms_pointer,
                            const double **feature_pointer,
                            const long long **cur_ncharges_pointer) {
  // Skip molecules current batch is not supposed to work with
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

void set_kernel_to_zero(double *kernel_matrix, long long start_mol1,
                        long long start_mol2, long long finish_mol1,
                        long long finish_mol2, long long nm2) {
  // Initially the kernel value is 0.
  double *kernel_pointer;
  for (long long i = start_mol1; i <= finish_mol1; i++) {
    kernel_pointer = &kernel_matrix[i * nm2 + start_mol2];
    for (long long j = start_mol2; j <= finish_mol2; j++) {
      *kernel_pointer = 0.0;
      kernel_pointer++;
    }
  }
}

void skip_molecules(const long long num_skipped_mols, const long long nfeatures,
                    const long long **cur_natoms_pointer,
                    const double **feature_pointer) {
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

    void
    local_dn_kernel(const double *q1, const double *q2, const long long *n1,
                    const long long *n2, const long long *ncharges1,
                    const long long *ncharges2, const double sigma_mult,
                    const long long nm1, const long long nm2,
                    const long long nfeatures, double *kernel_matrix) {

  // Distribute starting molecule ids according to the threads.
  long long start_mol1, start_mol2, finish_mol1, finish_mol2;
  set_total_thread_numbers_ids_local(&start_mol1, &start_mol2, &finish_mol1,
                                     &finish_mol2, n1, n2, nm1, nm2);
  // If some threads are ``extra'' terminate them.
  if (start_mol1 == -1 || start_mol2 == -1) {
    return;
  };
  long long mol1 = start_mol1;
  long long mol2 = start_mol2;

  // kernel_output_pointer is where atom-atom covariance is added as q2 is
  // cycled through. kernel_output_start_pointer is where kernel_output_pointer
  // is reset at completion of loop over q2.
  double *kernel_output_start_pointer = &kernel_matrix[mol1 * nm2 + mol2];
  double *kernel_output_pointer = kernel_output_start_pointer;

  const double *cur_q1_pointer = &q1[0];
  const long long *cur_n1_pointer = &n1[0];
  const long long *cur_ncharges1_pointer = &ncharges1[0];
  skip_molecules_wcharge(start_mol1, nfeatures, &cur_n1_pointer,
                         &cur_q1_pointer, &cur_ncharges1_pointer);

  const long long *n2_start_pointer = &n2[0];
  const double *q2_start_pointer = &q2[0];
  const long long *ncharges2_start_pointer = &ncharges2[0];
  skip_molecules_wcharge(start_mol2, nfeatures, &n2_start_pointer,
                         &q2_start_pointer, &ncharges2_start_pointer);

  const long long *cur_n2_pointer = n2_start_pointer;
  const double *cur_q2_pointer = q2_start_pointer;
  const long long *cur_ncharges2_pointer = ncharges2_start_pointer;

  // Atom counters.
  long long atom1 = 0; // counter for atom considered in mol1
  long long atom2 = 0; // counter for atom considered in mol2

  double temp_kernel_el;

  // Set all considered kernel elements to 0.0.
  set_kernel_to_zero(kernel_matrix, start_mol1, start_mol2, finish_mol1,
                     finish_mol2, nm2);

  bool skipped_pair, next_atom1, next_atom2;

  while (true) {
    skipped_pair = (*cur_ncharges1_pointer != *cur_ncharges2_pointer);
    if (not skipped_pair) {
      fetch_kernel_move_pointers(&temp_kernel_el, &cur_q1_pointer,
                                 &cur_q2_pointer, sigma_mult, nfeatures);
      *kernel_output_pointer += temp_kernel_el;
    };
    // proceed to calculate next atom
    atom2++;
    next_atom1 = false;
    next_atom2 = true;
    if (atom2 >= *cur_n2_pointer) { // move to next molecule
      atom2 = 0;
      next_atom1 = (mol2 == finish_mol2);
      if (next_atom1) { // move to next atom in molecule 1 and
                        // first molecule 2.
        atom1++;
        if (atom1 >= *cur_n1_pointer) {
          if (mol1 == finish_mol1) { // calculation finished
            return;
          }
          // move to next molecule 1
          atom1 = 0;
          mol1++;
          cur_n1_pointer++;
          kernel_output_start_pointer += nm2;
        };
        kernel_output_pointer = kernel_output_start_pointer;
        mol2 = start_mol2;
        cur_n2_pointer = n2_start_pointer;
        cur_q2_pointer = q2_start_pointer;
        cur_ncharges2_pointer = ncharges2_start_pointer;
        next_atom2 = false;
      } else {
        // move to next molecule 2.
        mol2++;
        cur_n2_pointer++;
        kernel_output_pointer++;
      }
    };
    if (next_atom1) {
      cur_ncharges1_pointer++;
      if (skipped_pair) {
        cur_q1_pointer += nfeatures;
      }
    } else {
      if (not skipped_pair) {
        cur_q1_pointer -= nfeatures;
      };
    };
    if (next_atom2) {
      cur_ncharges2_pointer++;
      if (skipped_pair) {
        cur_q2_pointer += nfeatures;
      }
    };
  }
}

extern "C" __global__ void
local_kernel(const double *q1, const double *q2, const long long *n1,
             const long long *n2, const double sigma_mult, const long long nm1,
             const long long nm2, const long long nfeatures,
             double *kernel_matrix) {

  // Distribute starting molecule ids according to the threads.
  long long start_mol1, start_mol2, finish_mol1, finish_mol2;
  set_total_thread_numbers_ids_local(&start_mol1, &start_mol2, &finish_mol1,
                                     &finish_mol2, n1, n2, nm1, nm2);
  // If some threads are ``extra'' terminate them.
  if (start_mol1 == -1 || start_mol2 == -1) {
    return;
  };
  long long mol1 = start_mol1;
  long long mol2 = start_mol2;

  // kernel_output_pointer is where atom-atom covariance is added as q2 is
  // cycled through. kernel_output_start_pointer is where kernel_output_pointer
  // is reset at completion of loop over q2.
  double *kernel_output_start_pointer = &kernel_matrix[mol1 * nm2 + mol2];
  double *kernel_output_pointer = kernel_output_start_pointer;

  const double *cur_q1_pointer = &q1[0];
  const long long *cur_n1_pointer = &n1[0];
  skip_molecules(start_mol1, nfeatures, &cur_n1_pointer, &cur_q1_pointer);

  const long long *n2_start_pointer = &n2[0];
  const double *q2_start_pointer = &q2[0];
  skip_molecules(start_mol2, nfeatures, &n2_start_pointer, &q2_start_pointer);

  const long long *cur_n2_pointer = n2_start_pointer;
  const double *cur_q2_pointer = q2_start_pointer;

  // Atom counters.
  long long atom1 = 0; // counter for atom considered in mol1
  long long atom2 = 0; // counter for atom considered in mol2

  double temp_kernel_el;

  // Set all considered kernel elements to 0.0.
  set_kernel_to_zero(kernel_matrix, start_mol1, start_mol2, finish_mol1,
                     finish_mol2, nm2);

  bool next_atom1;

  while (true) {
    fetch_kernel_move_pointers(&temp_kernel_el, &cur_q1_pointer,
                               &cur_q2_pointer, sigma_mult, nfeatures);
    *kernel_output_pointer += temp_kernel_el;
    // proceed to calculate next atom
    atom2++;
    next_atom1 = false;
    if (atom2 >= *cur_n2_pointer) { // move to next molecule
      atom2 = 0;
      next_atom1 = (mol2 == finish_mol2);
      if (next_atom1) { // move to next atom in molecule 1 and
                        // first molecule 2.
        atom1++;
        if (atom1 >= *cur_n1_pointer) {
          if (mol1 == finish_mol1) { // calculation finished
            return;
          }
          // move to next molecule 1
          atom1 = 0;
          mol1++;
          cur_n1_pointer++;
          kernel_output_start_pointer += nm2;
        };
        kernel_output_pointer = kernel_output_start_pointer;
        mol2 = start_mol2;
        cur_n2_pointer = n2_start_pointer;
        cur_q2_pointer = q2_start_pointer;
      } else {
        // move to next molecule 2.
        mol2++;
        cur_n2_pointer++;
        kernel_output_pointer++;
      }
    };
    if (not next_atom1) {
      cur_q1_pointer -= nfeatures;
    };
  }
}

void set_total_thread_numbers_ids_1D_global(
    long long nthreads, long long thread_id, long long nblocks,
    long long block_id, long long num_mols, long long *start_mol,
    long long *finish_mol) {
  long long tot_nthreads = nblocks * nthreads;
  long long tot_thread_id = nthreads * block_id + thread_id;
  long long workload = num_mols / tot_nthreads;
  long long workload_remainder = num_mols % tot_nthreads;
  if (tot_thread_id >= workload_remainder) {
    if (workload == 0) {
      *start_mol = -1;
      return;
    }
    *start_mol = workload_remainder * (workload + 1) +
                 (tot_thread_id - workload_remainder) * workload;
    *finish_mol = *start_mol + workload - 1;
  } else {
    *start_mol = tot_thread_id * (workload + 1);
    *finish_mol = *start_mol + workload;
  }
}

void set_total_thread_numbers_ids_global(long long *start_mol1,
                                         long long *start_mol2,
                                         long long *finish_mol1,
                                         long long *finish_mol2, long long nm1,
                                         long long nm2) {
  set_total_thread_numbers_ids_1D_global(blockDim.x, threadIdx.x, gridDim.x,
                                         blockIdx.x, nm1, start_mol1,
                                         finish_mol1);
  set_total_thread_numbers_ids_1D_global(blockDim.y, threadIdx.y, gridDim.y,
                                         blockIdx.y, nm2, start_mol2,
                                         finish_mol2);
}

extern "C" __global__ void kernel(const double *q1, const double *q2,
                                  const double sigma_mult, const long long nm1,
                                  const long long nm2,
                                  const long long nfeatures,
                                  double *kernel_matrix) {

  // Distribute starting molecule ids according to the threads.
  long long start_mol1, start_mol2, finish_mol1, finish_mol2;
  set_total_thread_numbers_ids_global(&start_mol1, &start_mol2, &finish_mol1,
                                      &finish_mol2, nm1, nm2);
  // If some threads are ``extra'' terminate them.
  if (start_mol1 == -1 || start_mol2 == -1) {
    return;
  }
  long long mol1 = start_mol1;
  long long mol2 = start_mol2;

  // kernel_output_pointer is where atom-atom covariance is added as q2 is
  // cycled through. kernel_output_start_pointer is where kernel_output_pointer
  // is reset at completion of loop over q2.
  double *kernel_output_pointer = &kernel_matrix[mol1 * nm2 + mol2];
  long long kernel_output_mol1_step = nm2 + start_mol2 - finish_mol2;
  // starting pointers where mols 2 are reset after cycling through them is done
  const double *q1_start_pointer = &q1[mol1 * nfeatures];
  const double *q2_start_pointer = &q2[mol2 * nfeatures];
  // Pointers towards relevant input, moved as mol1 and mol2 change.
  const double *cur_q1_pointer = q1_start_pointer;
  const double *cur_q2_pointer = q2_start_pointer;

  while (true) {
    fetch_kernel_move_pointers(kernel_output_pointer, &cur_q1_pointer,
                               &cur_q2_pointer, sigma_mult, nfeatures);
    if (mol2 == finish_mol2) {
      if (mol1 == finish_mol1) { // calculation finished
        return;
      }
      mol1 += 1;
      mol2 = start_mol2;
      cur_q2_pointer = q2_start_pointer;
      kernel_output_pointer += kernel_output_mol1_step;
    } else {
      mol2 += 1;
      kernel_output_pointer += 1;
      cur_q1_pointer -= nfeatures;
    };
  };
}
