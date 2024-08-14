#! /bin/bash

nvcc 0_naive.cu -std=c++11 -O3 && ./a.out
nv-nsight-cu-cli --metrics sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,smsp__sass_average_branch_targets_threads_uniform.pct,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio,smsp__sass_average_data_bytes_per_wavefront_mem_shared.pct,smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_membar_per_warp_active.pct ./a.out
