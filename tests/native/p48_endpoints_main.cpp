#include <assert.h>
#include <sched.h>
extern "C" void sbn3_endpoint4(int);
extern "C" void sbn3_endpoint5(int);
extern "C" void sbn3_endpoint6(int);
extern "C" void sbn3_endpoint7(int);
extern "C" void sbn3_endpoint8(int);
extern "C" void sbn3_endpoint9(int);
extern "C" void sbn3_endpoint10(int);
int main(int argc,char **){
    if(argc>1){cpu_set_t s;CPU_ZERO(&s);CPU_SET(2,&s);assert(sched_setaffinity(0,sizeof s,&s)==0);}
    sbn3_endpoint4(argc>1);sbn3_endpoint5(argc>1);sbn3_endpoint6(argc>1);sbn3_endpoint7(argc>1);sbn3_endpoint8(argc>1);sbn3_endpoint9(argc>1);sbn3_endpoint10(argc>1);
}
