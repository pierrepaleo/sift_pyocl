
void __kernel interpolate(image3d_t volume,
                          float* img,
                          int img_width,
                          int img_height,
                          float3* point,
                          float3* norm)
                       {
int pos_x = get_global_id(0);
int pos_y = get_global_id(1);
if (pos_x>=img_width)||(pos_y>img_height)
    return;
float center_x = get_global_id(0)/2.0f;
float center_y = get_global_id(1)/2.0f;
float3 n_norm = normalize(norm[0]);
float3 u_norm, v_norm
float nx = n_norm.x,
      ny = n_norm.y,
      nz = n_norm.z;
float ax = abs(nx),
      ay = abs(ny),
      az = abs(nz);

if (ax>=az) && (ay>=az)       //z smallest
	u_norm = (float3)( -ny, nx, 0.0f);
else if  (ax>=ay) && (az>=ay) //y smallest
    u_norm = (float3)( -nz, 0.0f, nx);
else if  (ay>=ax) && (az>=ax) //x smallest
    u_norm = (float3)( 0.0f, -nz, ny);

//define 2 largest components a,b,c
//create  orthogonal vector -b a 0
// create third vector by cross product
//use them to

}
