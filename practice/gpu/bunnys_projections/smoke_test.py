"""Offline analytic projection / occlusion and CPU-CUDA equivalence checks."""
def check_renderers():
    from stanford_scene import np,cp,GPU_AVAILABLE,render_instances_cpu,INSTANCE_KERNEL
    points=np.array([[0,0,2],[0,0,1],[0,0,-1],[100,0,1]],np.float32)
    offsets=np.zeros((1,3),np.float32);R=np.eye(3,dtype=np.float32);eye=np.zeros(3,np.float32)
    cpu=render_instances_cpu(points,offsets,R,eye)
    assert np.isfinite(cpu).sum()==1 and cpu[240,320]==1, 'Projection/z-buffer/near-plane failed'
    if GPU_AVAILABLE:
        kernel=cp.RawKernel(INSTANCE_KERNEL,'project_depth',options=('--fmad=false',))
        depth=cp.full((480,640),cp.inf,cp.float32)
        kernel((1,),(256,),(cp.asarray(points),cp.asarray(offsets),np.int32(4),cp.asarray(R),cp.asarray(eye),depth,np.int32(4),np.int32(640),np.int32(480),np.float32(500)))
        assert np.array_equal(cpu,cp.asnumpy(depth)), 'CPU/GPU mismatch'
if __name__=='__main__':
    check_renderers()
    print('PASS')
