//CL//
    #define GROUP_SIZE 64
    #define READ_AND_MAP(i) (mmc_from_scalar(x[i]))
    #define REDUCE(a, b) (agg_mmc(a, b))

        #pragma OPENCL EXTENSION cl_khr_fp64: enable
        #define PYOPENCL_DEFINE_CDOUBLE

    #include <pyopencl-complex.h>

    typedef struct {
  int cur_min;
  int cur_max;
  int pad;
} minmax_collector;

//CL//
minmax_collector mmc_neutral()
{
// FIXME: needs infinity literal in real use, ok here
minmax_collector result;
result.cur_min = 1<<30;
result.cur_max = -(1<<30);
return result;
}
minmax_collector mmc_from_scalar(float x)
{
minmax_collector result;
result.cur_min = x;
result.cur_max = x;
return result;
}
minmax_collector agg_mmc(minmax_collector a, minmax_collector b)
{
minmax_collector result = a;
if (b.cur_min < result.cur_min)
result.cur_min = b.cur_min;
if (b.cur_max > result.cur_max)
result.cur_max = b.cur_max;
return result;
}


    typedef minmax_collector out_type;

    __kernel void reduce_kernel_stage1(
      __global out_type *out, __global int *x,
      unsigned int seq_count, unsigned int n)
    {
        __local out_type ldata[GROUP_SIZE];

        unsigned int lid = get_local_id(0);

        unsigned int i = get_group_id(0)*GROUP_SIZE*seq_count + lid;

        out_type acc = mmc_neutral();
        for (unsigned s = 0; s < seq_count; ++s)
        {
          if (i >= n)
            break;
          acc = REDUCE(acc, READ_AND_MAP(i));

          i += GROUP_SIZE;
        }

        ldata[lid] = acc;

        

            barrier(CLK_LOCAL_MEM_FENCE);

            

            if (lid < 32)
            {
                ldata[lid] = REDUCE(
                  ldata[lid],
                  ldata[lid + 32]);
            }

            



            barrier(CLK_LOCAL_MEM_FENCE);

            if (lid < 32)
            {
                __local volatile out_type *lvdata = ldata;
                    

                    lvdata[lid] = REDUCE(
                      lvdata[lid],
                      lvdata[lid + 16]);

                    

                    

                    lvdata[lid] = REDUCE(
                      lvdata[lid],
                      lvdata[lid + 8]);

                    

                    

                    lvdata[lid] = REDUCE(
                      lvdata[lid],
                      lvdata[lid + 4]);

                    

                    

                    lvdata[lid] = REDUCE(
                      lvdata[lid],
                      lvdata[lid + 2]);

                    

                    

                    lvdata[lid] = REDUCE(
                      lvdata[lid],
                      lvdata[lid + 1]);

                    


            }

        if (lid == 0) out[get_group_id(0)] = ldata[0];
    }
