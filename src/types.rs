use ash::vk;
use std::sync::Arc;

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub struct TextureDesc {
    pub usage: vk::ImageUsageFlags,
    pub format: vk::Format,
    pub extent: vk::Extent3D
}

#[derive(Clone)]
pub struct FrameGraphTexture {
    pub(crate) last_access: TextureAccess,
    pub(crate) subresource_range: vk::ImageSubresourceRange,
    pub(crate) allocation: Option<Arc<GpuAllocation>>,
    pub(crate) image: vk::Image,
}

#[derive(Clone)]
pub struct FrameGraphBuffer {
    pub(crate) last_access: BufferAccess,
    pub(crate) allocation: Option<Arc<GpuAllocation>>,
    pub(crate) image: vk::Buffer,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum TextureAccess {
    ColorWrite,
    DepthWrite,
    DepthRead,
    VertexRead,
    FragmentRead,
    ComputeRead,
    ComputeWrite,
    TransferSrc,
    TransferDst,
    Present,
    Undefined,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum BufferAccess {
    ComputeWrite,
    ComputeRead
}

pub(crate) enum FrameGraphResource {
    Texture(FrameGraphTexture),
    Buffer(FrameGraphBuffer)
}

pub(crate) enum ResourceState {
    Transient(TextureDesc),
    Imported(FrameGraphTexture)
}

impl From<FrameGraphTexture> for FrameGraphResource {
    fn from(value: FrameGraphTexture) -> Self {
        Self::Texture(value)
    }
}

impl From<FrameGraphBuffer> for FrameGraphResource {
    fn from(value: FrameGraphBuffer) -> Self {
        Self::Buffer(value)
    }
}

#[cfg(all(feature = "gpu-allocator", feature = "vk-mem"))]
compile_error!("Only one feature \"gpu-allocator\" or \"vk-mem\" must be enabled for this crate");

#[cfg(not(any(feature = "gpu-allocator", feature = "vk-mem")))]
compile_error!("Feature \"gpu-allocator\" or \"vk-mem\" must be enabled for this crate");

#[cfg(all(feature = "parking_lot", feature = "vk-mem"))]
compile_error!("Feature \"parking_lot\" with \"vk-mem\" incompatible for this crate");

#[cfg(all(not(feature = "parking_lot"), feature = "gpu-allocator"))]
pub(crate) type GpuAllocator = Arc<std::sync::Mutex<gpu_allocator::vulkan::Allocator>>;

#[cfg(all(feature = "parking_lot", feature = "gpu-allocator"))]
pub(crate) type GpuAllocator = Arc<parking_lot::Mutex<gpu_allocator::vulkan::Allocator>>;

#[cfg(feature = "gpu-allocator")]
type GpuAllocation = gpu_allocator::vulkan::Allocation;

#[cfg(feature = "vk-mem")]
pub(crate) type GpuAllocator = Arc<vk_mem::Allocator>;

#[cfg(feature = "vk-mem")]
type GpuAllocation = vk_mem::Allocation;