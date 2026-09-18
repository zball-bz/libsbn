/* Imported from libsbn/include/sbn/detail/u52/addsub_nc.inc; arithmetic preserved, private namespace and prepared Frame adapter. */
inline static void _MPN_ADDSUB_N(sb_pvec r, sb_cpvec a, sb_cpvec b, uint64_t n) {
  // add/sub but non-canonical; 
  sb_vec r0, a0, b0;
  uint64_t cnt = n >> 3;
  const uint64_t tail = n & 7;
  const __mmask8 m = u52_limb_mask(tail);
#pragma GCC unroll 4
  for(; cnt; --cnt){
    a0 = sb_load(a++);
    b0 = sb_load(b++);
    flex_add(r0, a0, b0);
    sb_store(r++, r0);
  }
  if(tail){
    a0 = sb_load(a,m);
    b0 = sb_load(b,m);
    flex_add(r0, a0, b0);
    sb_store(r, r0, m);
  }
}
