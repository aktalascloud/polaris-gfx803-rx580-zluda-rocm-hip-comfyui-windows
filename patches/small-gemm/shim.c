/* SPDX-License-Identifier: GPL-3.0-only */
typedef void* handle_t;
typedef long long stride_t;
typedef unsigned int uint_t;
typedef int status_t;
int _fltused = 0;
typedef int (__stdcall *once_callback_t)(void*, void*, void**);
__declspec(dllimport) void* __stdcall GetModuleHandleA(const char*);
__declspec(dllimport) void* __stdcall GetProcAddress(void*, const char*);
__declspec(dllimport) int __stdcall GetModuleHandleExA(uint_t, const char*, void**);
__declspec(dllimport) uint_t __stdcall GetModuleFileNameA(void*, char*, uint_t);
__declspec(dllimport) int __stdcall InitOnceExecuteOnce(void*, once_callback_t, void*, void**);
__declspec(dllimport) void* __stdcall LoadLibraryExA(const char*, void*, uint_t);

static void* kernel_once;
static void* kernel_function;
static void* kernel_context;
static int (*hip_launch)(void*, uint_t, uint_t, uint_t, uint_t, uint_t, uint_t, uint_t, void*, void**, void**);

static void* backend_once;
static void* backend_module;
static int __stdcall initialize_backend(void* once, void* unused, void** context)
{
    void* self = 0;
    char path[1024];
    const char filename[] = "rocblas_backend.dll";
    uint_t length, start = 0, index;
    (void)once; (void)unused; (void)context;
    if (!GetModuleHandleExA(6, (const char*)&initialize_backend, &self)) return 0;
    length = GetModuleFileNameA(self, path, sizeof(path));
    if (!length || length >= sizeof(path)) return 0;
    for (index=0; index<length; ++index) if (path[index]=='\\' || path[index]=='/') start=index+1;
    if (start+sizeof(filename)>sizeof(path)) return 0;
    for (index=0; index<sizeof(filename); ++index) path[start+index]=filename[index];
    backend_module=LoadLibraryExA(path, 0, 0x1100);
    return backend_module != 0;
}
static void* symbol(const char* name)
{
    if (!InitOnceExecuteOnce(&backend_once, initialize_backend, 0, 0)) return 0;
    return GetProcAddress(backend_module, name);
}

/* The legacy Windows driver does not implement hipDeviceAttributeGcnArch.
   Restrict the code object to recognized Polaris RX names, not RDNA RX 5700. */
static int polaris_name(const char* name)
{
    uint_t i;
    for (i=0; name[i] && i<240; ++i) {
        const char* p;
        if (name[i]!='R' || name[i+1]!='X') continue;
        p=name+i+2; while (*p==' ') ++p;
        if ((p[0]=='4' && (p[1]=='7' || p[1]=='8')) ||
            (p[0]=='5' && (p[1]=='7' || p[1]=='8' || p[1]=='9'))) {
            if (p[2]=='0' && !(p[3]>='0' && p[3]<='9')) return 1;
        }
    }
    return 0;
}

static int __stdcall initialize_kernel(void* once, void* unused, void** context)
{
    void* self = 0;
    char path[1024];
    const char filename[] = "rocblas_small_gemm.hsaco";
    uint_t length, start, index;
    void* hip = GetModuleHandleA("amdhip64.dll");
    int (*module_load)(void**, const char*) = (int (*)(void**, const char*))GetProcAddress(hip, "hipModuleLoad");
    int (*get_function)(void**, void*, const char*) = (int (*)(void**, void*, const char*))GetProcAddress(hip, "hipModuleGetFunction");
    void* module = 0;
    int (*get_context)(void**) = (int (*)(void**))GetProcAddress(hip, "hipCtxGetCurrent");
    (void)once; (void)unused; (void)context;
    if (!module_load || !get_function || !get_context || get_context(&kernel_context) || !kernel_context) return 0;
    hip_launch = (int (*)(void*, uint_t, uint_t, uint_t, uint_t, uint_t, uint_t, uint_t, void*, void**, void**))GetProcAddress(hip, "hipModuleLaunchKernel");
    if (!hip_launch || !GetModuleHandleExA(6, (const char*)&initialize_kernel, &self)) return 0;
    length = GetModuleFileNameA(self, path, sizeof(path));
    if (!length || length >= sizeof(path)) return 0;
    start = 0;
    for (index = 0; index < length; ++index) if (path[index] == '\\' || path[index] == '/') start = index + 1;
    if (start + sizeof(filename) > sizeof(path)) return 0;
    for (index = 0; index < sizeof(filename); ++index) path[start + index] = filename[index];
    if (module_load(&module, path)) return 0;
    if (get_function(&kernel_function, module, "gfx803_small_sgemm")) return 0;
    return 1;
}

/* A native HIP source kernel is used only for small FP32, host-scalar GEMMs.
   Other dtypes, explicit solution selection, graph capture and other devices
   keep the original rocBLAS implementation and ABI. */
static status_t try_small(handle_t handle, int ta, int tb, int m, int n, int k,
    const float* alpha, const float* a, int lda, stride_t sa,
    const float* b, int ldb, stride_t sb, const float* beta,
    const float* c, int ldc, stride_t sc, float* d, int ldd, stride_t sd, int batches)
{
    int mode = -1, device = -1, capturing = 0, transa, transb;
    float av, bv;
    void* stream = 0;
    void* hip;
    int (*get_mode)(handle_t, int*);
    int (*get_stream)(handle_t, void**);
    unsigned char (*is_size_query)(handle_t);
    int (*get_device)(int*);
    int (*is_capturing)(void*, int*);
    int (*get_name)(char*, int, int);
    char device_name[256] = {0};
    int (*get_context)(void**);
    void* current_context = 0;
    void* parameters[19];
    if (!handle || m < 1 || m > 8 || n < 1 || n > 8 || k < 1 || k > 8 || batches < 1) return -1;
    if (ta < 111 || ta > 113 || tb < 111 || tb > 113) return -1;
    transa = ta != 111; transb = tb != 111;
    if (lda < (transa ? k : m) || ldb < (transb ? n : k) || ldc < m || ldd < m || !alpha || !beta || !d) return -1;
    get_mode = (int (*)(handle_t, int*))symbol("rocblas_get_pointer_mode");
    get_stream = (int (*)(handle_t, void**))symbol("rocblas_get_stream");
    is_size_query = (unsigned char (*)(handle_t))symbol("rocblas_is_device_memory_size_query");
    if (!get_mode || !get_stream || !is_size_query || is_size_query(handle)) return -1;
    if (get_mode(handle, &mode) || mode != 0 || get_stream(handle, &stream)) return -1;
    av = *alpha; bv = *beta;
    if ((av != 0.0f && (!a || !b)) || (bv != 0.0f && !c)) return -1;
    hip = GetModuleHandleA("amdhip64.dll");
    get_device = (int (*)(int*))GetProcAddress(hip, "hipGetDevice");
    is_capturing = (int (*)(void*, int*))GetProcAddress(hip, "hipStreamIsCapturing");
    get_name = (int (*)(char*, int, int))GetProcAddress(hip, "hipDeviceGetName");
    get_context = (int (*)(void**))GetProcAddress(hip, "hipCtxGetCurrent");
    if (!get_device || get_device(&device) || device != 0) return -1;
    if (!get_name || get_name(device_name, sizeof(device_name), device) || !polaris_name(device_name)) return -1;
    if (!is_capturing || is_capturing(stream, &capturing) || capturing) return -1;
    if (!InitOnceExecuteOnce(&kernel_once, initialize_kernel, 0, 0)) return -1;
    /* A module belongs to its initial context. Never reuse it in a new one. */
    if (!get_context || get_context(&current_context) || current_context != kernel_context) return -1;
    parameters[0] = &a; parameters[1] = &b; parameters[2] = &c; parameters[3] = &d;
    parameters[4] = &m; parameters[5] = &n; parameters[6] = &k;
    parameters[7] = &lda; parameters[8] = &ldb; parameters[9] = &ldc; parameters[10] = &ldd;
    parameters[11] = &sa; parameters[12] = &sb; parameters[13] = &sc; parameters[14] = &sd;
    parameters[15] = &transa; parameters[16] = &transb; parameters[17] = &av; parameters[18] = &bv;
    return hip_launch(kernel_function, (uint_t)batches, 1, 1, 64, 1, 1, 0, stream, parameters, 0) ? 6 : 0;
}

#define SGEMM_ARGS handle_t h,int ta,int tb,int m,int n,int k,const float* alpha,const float* a,int lda,const float* b,int ldb,const float* beta,float* c,int ldc
status_t rocblas_sgemm(SGEMM_ARGS)
{
    status_t result = try_small(h,ta,tb,m,n,k,alpha,a,lda,0,b,ldb,0,beta,c,ldc,0,c,ldc,0,1);
    if (result >= 0) return result;
    { status_t (*fn)(SGEMM_ARGS) = (status_t (*)(SGEMM_ARGS))symbol("rocblas_sgemm");
      return fn ? fn(h,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc) : 6; }
}

#define STRIDED_ARGS handle_t h,int ta,int tb,int m,int n,int k,const float* alpha,const float* a,int lda,stride_t sa,const float* b,int ldb,stride_t sb,const float* beta,float* c,int ldc,stride_t sc,int batches
status_t rocblas_sgemm_strided_batched(STRIDED_ARGS)
{
    status_t result = try_small(h,ta,tb,m,n,k,alpha,a,lda,sa,b,ldb,sb,beta,c,ldc,sc,c,ldc,sc,batches);
    if (result >= 0) return result;
    { status_t (*fn)(STRIDED_ARGS) = (status_t (*)(STRIDED_ARGS))symbol("rocblas_sgemm_strided_batched");
      return fn ? fn(h,ta,tb,m,n,k,alpha,a,lda,sa,b,ldb,sb,beta,c,ldc,sc,batches) : 6; }
}

#define EX_ARGS handle_t h,int ta,int tb,int m,int n,int k,const void* alpha,const void* a,int at,int lda,const void* b,int bt,int ldb,const void* beta,const void* c,int ct,int ldc,void* d,int dt,int ldd,int compute,int algo,int solution,uint_t flags
status_t rocblas_gemm_ex(EX_ARGS)
{
    if (at==151 && bt==151 && ct==151 && dt==151 && compute==151 && algo==0 && solution==0 && flags==0) {
        status_t result=try_small(h,ta,tb,m,n,k,alpha,a,lda,0,b,ldb,0,beta,c,ldc,0,d,ldd,0,1);
        if(result>=0) return result;
    }
    { status_t (*fn)(EX_ARGS) = (status_t (*)(EX_ARGS))symbol("rocblas_gemm_ex");
      return fn ? fn(h,ta,tb,m,n,k,alpha,a,at,lda,b,bt,ldb,beta,c,ct,ldc,d,dt,ldd,compute,algo,solution,flags) : 6; }
}

#define STRIDED_EX_ARGS handle_t h,int ta,int tb,int m,int n,int k,const void* alpha,const void* a,int at,int lda,stride_t sa,const void* b,int bt,int ldb,stride_t sb,const void* beta,const void* c,int ct,int ldc,stride_t sc,void* d,int dt,int ldd,stride_t sd,int batches,int compute,int algo,int solution,uint_t flags
status_t rocblas_gemm_strided_batched_ex(STRIDED_EX_ARGS)
{
    if (at==151 && bt==151 && ct==151 && dt==151 && compute==151 && algo==0 && solution==0 && flags==0) {
        status_t result=try_small(h,ta,tb,m,n,k,alpha,a,lda,sa,b,ldb,sb,beta,c,ldc,sc,d,ldd,sd,batches);
        if(result>=0) return result;
    }
    { status_t (*fn)(STRIDED_EX_ARGS) = (status_t (*)(STRIDED_EX_ARGS))symbol("rocblas_gemm_strided_batched_ex");
      return fn ? fn(h,ta,tb,m,n,k,alpha,a,at,lda,sa,b,bt,ldb,sb,beta,c,ct,ldc,sc,d,dt,ldd,sd,batches,compute,algo,solution,flags) : 6; }
}

int rocblas_windows_small_gemm_patch_version(void) { return 1; }
