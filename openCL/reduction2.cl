

__kernel void reduce_kernel_stage2(
  __global float *out, __global const float *pyopencl_reduction_inp, __global const float *in,
  unsigned int seq_count, unsigned int size)
{
    __local float ldata[GROUP_SIZE];

    unsigned int lid = get_local_id(0);

    unsigned int i = get_group_id(0)*GROUP_SIZE*seq_count + lid;

    float acc = 0;
    for (unsigned s = 0; s < seq_count; ++s)
    {
      if (i >= size)
        break;
      acc = REDUCE(acc, READ_AND_MAP(i));

      i += GROUP_SIZE;
    }

    ldata[lid] = acc;

    

        barrier(CLK_LOCAL_MEM_FENCE);

        

        if (lid < 512)
        {
            ldata[lid] = REDUCE(
              ldata[lid],
              ldata[lid + 512]);
        }

        

        barrier(CLK_LOCAL_MEM_FENCE);

        

        if (lid < 256)
        {
            ldata[lid] = REDUCE(
              ldata[lid],
              ldata[lid + 256]);
        }

        

        barrier(CLK_LOCAL_MEM_FENCE);

        

        if (lid < 128)
        {
            ldata[lid] = REDUCE(
              ldata[lid],
              ldata[lid + 128]);
        }

        

        barrier(CLK_LOCAL_MEM_FENCE);

        

        if (lid < 64)
        {
            ldata[lid] = REDUCE(
              ldata[lid],
              ldata[lid + 64]);
        }

        

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
            __local volatile float *lvdata = ldata;
                

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
