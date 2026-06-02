use ash::vk;
use gpu_allocator::{MemoryLocation, vulkan::*};
use std::{collections::HashMap, marker::PhantomData, sync::{Arc, Mutex}};
use slotmap::{SlotMap, new_key_type};

mod error;
use error::*;

new_key_type! {
    struct Key;
}

#[derive(Clone, Copy)]
pub struct Handle<T> {
    key: Key,
    _marker: PhantomData<T>
}

impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
    fn ne(&self, other: &Self) -> bool {
        self.key != other.key
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub struct TextureDesc {
    pub usage: vk::ImageUsageFlags,
    pub format: vk::Format,
    pub extent: vk::Extent3D
}

#[derive(Clone)]
pub struct FrameGraphTexture {
    last_access: vk_sync::AccessType,
    allocation: Option<Arc<Allocation>>,
    image: vk::Image,
}

pub struct FrameScope<'a> {
    graph: &'a mut FrameGraph,
    /// current frame handle -> texture
    texture_handles: SlotMap<Key, FrameGraphTexture>,
    imported_textures: Vec<FrameGraphTexture>,
    exported_textures: Vec<(Handle<FrameGraphTexture>, vk_sync::AccessType)>,
    compiled_passes: Vec<CompiledPass<'a>>,
    execution_order: Vec<usize>,
    pass_descs: Vec<Pass<'a>>,
}

impl<'a> FrameScope<'a> {
    pub fn new(graph: &'a mut FrameGraph) -> Self {
        Self { 
            graph,
            texture_handles: SlotMap::with_key(), 
            imported_textures: vec![], 
            exported_textures: vec![], 
            compiled_passes: vec![], 
            execution_order: vec![], 
            pass_descs: vec![]
        }
    }
    
    pub fn create(&mut self, desc: TextureDesc) -> Handle<FrameGraphTexture> {
        // next image
        let idx = (self.graph.current_frame + 1) % self.graph.frame_in_flight;

        // transient textures for current frame
        let transient_textures = self.graph.transient_textures.entry(idx)
            .or_insert(HashMap::new());

        // all textures for this desc
        let textures = transient_textures.entry(desc).or_insert(vec![]);

        for (in_use, tex) in textures.iter_mut() {
            if !*in_use {
                *in_use = true;
                let key = self.texture_handles.insert(tex.clone());
                return Handle { 
                    key, 
                    _marker: PhantomData 
                };
            }
        }

        let image = unsafe {

            let create_info: vk::ImageCreateInfo<'_> = vk::ImageCreateInfo::default()
                .format(desc.format)
                .extent(desc.extent)
                .array_layers(1)
                .image_type(vk::ImageType::TYPE_2D)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(desc.usage);

            self.graph.device
                .create_image(&create_info, None)
                .unwrap()
        };

        let requirements = unsafe { self.graph.device.get_image_memory_requirements(image) };

        let mut allocator = self.graph.allocator.try_lock().unwrap();

        let alloc_desc = AllocationCreateDesc {
            name: "Transient Texture",
            location: MemoryLocation::GpuOnly,
            requirements,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };

        let allocation = allocator.allocate(&alloc_desc).unwrap();

        unsafe { 
            self.graph.device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
                .unwrap() 
        };

        let frame_texture = FrameGraphTexture {
            last_access: vk_sync::AccessType::Nothing,
            allocation: Some(Arc::new(allocation)),
            image
        };

        textures.push((true, frame_texture.clone()));

        let key = self.texture_handles.insert(frame_texture);

        Handle { 
            key,
            _marker: PhantomData 
        }
    }

    pub fn import(&mut self, image: vk::Image, current_access: vk_sync::AccessType) -> Handle<FrameGraphTexture> {
        let frame_texture = FrameGraphTexture {
            last_access: current_access,
            allocation: None,
            image
        };

        self.imported_textures.push(frame_texture.clone());
        let key = self.texture_handles.insert(frame_texture);

        Handle { 
            key, 
            _marker: PhantomData 
        }
    }

    pub fn export(&mut self, handle: Handle<FrameGraphTexture>, access: vk_sync::AccessType) -> vk::Image {
        let tex = self.texture_handles.get(handle.key).unwrap();
        self.exported_textures.push((handle, access));
        tex.image
    }

    fn topological_sort(dependencies: &[Vec<usize>]) -> Vec<usize> {
        let n = dependencies.len();

        let mut in_degree: Vec<usize> = dependencies.iter().map(|d| d.len()).collect();

        let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut result = Vec::with_capacity(n);

        while let Some(node) = queue.pop() {
            result.push(node);

            for j in 0..n {
                if dependencies[j].contains(&node) {
                    in_degree[j] -= 1;
                    if in_degree[j] == 0 {
                        queue.push(j);
                    }
                }
            }
        }

        result
    }
    
    pub fn sorting_passes(passes: &Vec<Pass>) -> Vec<usize> {
        let mut dependencies: Vec<Vec<usize>> = vec![vec![]; passes.len()];

        for (i, pass_a) in passes.iter().enumerate() {
            let writes = &pass_a.writes;
            for (j, pass_b) in passes.iter().enumerate() {
                if i != j && pass_b.reads.iter().any(|r| writes.contains(r)) {
                    dependencies[j].push(i);
                }
            }
        }

        let sorted_indices = Self::topological_sort(&dependencies);
        sorted_indices
    }

    pub fn compile(&mut self) {

        let indices = Self::sorting_passes(&self.pass_descs);
        self.execution_order = indices.clone();

        let mut compiled_passes = vec![];
        let mut export_image_barriers = vec![];

        for i in indices {
            let pass = &mut self.pass_descs[i];

            let mut image_barriers = vec![];

            for (handle, required_access) in &pass.reads {
                let tex = self.texture_handles.get_mut(handle.key).unwrap();

                let (src_stage, src_access, src_layout) = access_to_sync2(tex.last_access);
                let (dst_stage, dst_access, dst_layout) = access_to_sync2(*required_access);

                if src_layout == dst_layout && tex.last_access == *required_access {
                    continue;
                }

                image_barriers.push(
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(src_stage)
                        .src_access_mask(src_access)
                        .dst_stage_mask(dst_stage)
                        .dst_access_mask(dst_access)
                        .old_layout(src_layout)
                        .new_layout(dst_layout)
                        .image(tex.image)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .base_mip_level(0)
                                .level_count(1)
                                .base_array_layer(0)
                                .layer_count(1),
                        )
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
                );

                tex.last_access = *required_access;
            }

            for (handle, required_access) in &pass.writes {
                let tex = self.texture_handles.get_mut(handle.key).unwrap();

                let (src_stage, src_access, src_layout) = access_to_sync2(tex.last_access);
                let (dst_stage, dst_access, dst_layout) = access_to_sync2(*required_access);

                if src_layout == dst_layout && tex.last_access == *required_access {
                    continue;
                }

                image_barriers.push(
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(src_stage)
                        .src_access_mask(src_access)
                        .dst_stage_mask(dst_stage)
                        .dst_access_mask(dst_access)
                        .old_layout(src_layout)
                        .new_layout(dst_layout)
                        .image(tex.image)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .base_mip_level(0)
                                .level_count(1)
                                .base_array_layer(0)
                                .layer_count(1),
                        )
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
                );

                tex.last_access = *required_access;
            }

            for (handle, required_access) in &self.exported_textures {
                let tex = self.texture_handles.get_mut(handle.key).unwrap();

                let (src_stage, src_access, src_layout) = access_to_sync2(tex.last_access);
                let (dst_stage, dst_access, dst_layout) = access_to_sync2(*required_access);

                if src_layout == dst_layout && tex.last_access == *required_access {
                    continue;
                }

                export_image_barriers.push(
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(src_stage)
                        .src_access_mask(src_access)
                        .dst_stage_mask(dst_stage)
                        .dst_access_mask(dst_access)
                        .old_layout(src_layout)
                        .new_layout(dst_layout)
                        .image(tex.image)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .base_mip_level(0)
                                .level_count(1)
                                .base_array_layer(0)
                                .layer_count(1),
                        )
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED),
                );

                tex.last_access = *required_access;
            }

            compiled_passes.push(CompiledPass { 
                before_execute: Some(Box::new(move |device, cmd, _| {
                    unsafe { 
                        let dep_info = vk::DependencyInfoKHR::default()
                            .image_memory_barriers(&image_barriers);

                        device.cmd_pipeline_barrier2(cmd, &dep_info);
                    };
                })), 
                execute: pass.callback.take(), 
                after_execute: None
            });
        }

        if !export_image_barriers.is_empty() {
            let export_idx = compiled_passes.len();
            compiled_passes.push(CompiledPass {
                before_execute: None,
                execute: None,
                after_execute: Some(Box::new(move |device, cmd, _| unsafe {
                    let dep_info = vk::DependencyInfo::default()
                        .image_memory_barriers(&export_image_barriers);
                    device.cmd_pipeline_barrier2(cmd, &dep_info);
                })),
            });
            self.execution_order.push(export_idx);
        }

        self.pass_descs.clear();
        self.compiled_passes = compiled_passes;
    }

    pub fn execute(&mut self, cmd: vk::CommandBuffer) {
        for index in &self.execution_order {
            let pass = &mut self.compiled_passes[*index];

            if let Some(before_execute) = pass.before_execute.take() {
                before_execute(&self.graph.device, cmd, PassResources { resources: &self.texture_handles });
            }

            if let Some(execute) = pass.execute.take() {
                execute(&self.graph.device, cmd, PassResources { resources: &self.texture_handles });
            }

            if let Some(after_execute) = pass.after_execute.take() {
                after_execute(&self.graph.device, cmd, PassResources { resources: &self.texture_handles });
            }
        }

        for (_, frame_textures) in &mut self.graph.transient_textures {
            for (_, textures) in frame_textures.iter_mut() {
                for j in textures {
                    j.0 = false;
                }
            }
        }

        self.graph.current_frame += 1;
        self.texture_handles.clear();
    }

    pub fn add_pass(&mut self, pass: Pass<'a>) {
        self.pass_descs.push(pass);
    }
}

/// Required Sync2
pub struct FrameGraph {
    device: ash::Device,
    allocator: Arc<Mutex<Allocator>>,
    current_frame: usize,
    frame_in_flight: usize,
    // index of frame -> list of free transient textures
    transient_textures: HashMap<usize, HashMap<TextureDesc, Vec<(bool, FrameGraphTexture)>>>,
}

impl Drop for FrameGraph {
    fn drop(&mut self) {
        // We are waiting for the gpu to finish working with all the images
        match unsafe { self.device.device_wait_idle() } {
            Ok(_) => {
                // Lock gpu allocator
                match self.allocator.lock() {
                    Ok(mut guard) => {
                        let _ = self.transient_textures
                            .drain()
                            .map(|(_, mut frame_textures)| {
                                let _ = frame_textures.drain()
                                    .map(|(_, mut textures)| {
                                        let _ = textures.drain(..)
                                            .map(|(_, tex)| {
                                                if let Some(allocation) = tex.allocation {
                                                    if let Some(inner_allocation) = Arc::into_inner(allocation) {
                                                        let _ = guard.free(inner_allocation);
                                                    } else {
                                                        log::error!("FrameScope still alive when dropping FrameGraph - memory leaks")
                                                    }
                                                }
                                            });
                                    });   
                            });
                    },
                    Err(err) => {
                        log::error!("Failed to lock allocator mutex: {:?}", err)
                    }
                }
            },
            Err(err) => {
                log::error!("Failed to wait for device idle: {:?}", err)
            }
        }
    }
}

pub struct FrameGraphCreateDesc {
    pub allocator: Arc<Mutex<Allocator>>,
    pub frame_in_flight: usize,
    pub device: ash::Device,
}

impl FrameGraph {
    pub fn new(desc: &FrameGraphCreateDesc) -> Self {
        Self {
            frame_in_flight: desc.frame_in_flight,
            current_frame: 0,
            allocator: desc.allocator.clone(),
            transient_textures: HashMap::new(),
            device: desc.device.clone(),
        }
    }
}

struct CompiledPass<'a> {
    before_execute: Option<Box<dyn FnOnce(&ash::Device, vk::CommandBuffer, PassResources) + 'a>>,
    execute: Option<Box<dyn FnOnce(&ash::Device, vk::CommandBuffer, PassResources) + 'a>>,
    after_execute: Option<Box<dyn FnOnce(&ash::Device, vk::CommandBuffer, PassResources) + 'a>>
}

pub struct PassResources<'a> {
    resources: &'a SlotMap<Key, FrameGraphTexture>
}

impl<'a> PassResources<'a> {
    pub fn get(&self, handle: Handle<FrameGraphTexture>) -> vk::Image {
        let tex = self.resources.get(handle.key).unwrap();
        tex.image
    }
}

pub struct Pass<'a> {
    name: String,
    reads: Vec<(Handle<FrameGraphTexture>, vk_sync::AccessType)>,
    writes: Vec<(Handle<FrameGraphTexture>, vk_sync::AccessType)>,
    callback: Option<Box<dyn FnOnce(&ash::Device, vk::CommandBuffer, PassResources) + 'a>>
}

impl std::fmt::Debug for Pass<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pass")
            .field("name", &self.name)
            .field("reads", &self.reads.iter().map(|(h, a)| (h.key, a)).collect::<Vec<_>>())
            .field("writes", &self.writes.iter().map(|(h, a)| (h.key, a)).collect::<Vec<_>>())
            .finish()
    }
}

impl<'a> Pass<'a> {
    pub fn new<S: Into<String>>(name: S) -> Self {
        Self {
            callback: None,
            reads: vec![],
            writes: vec![],
            name: name.into()
        }
    }

    pub fn color_attachment(mut self, handle: Handle<FrameGraphTexture>, load: vk::AttachmentLoadOp, store: vk::AttachmentStoreOp) -> Self {
        self
    }

    pub fn depth_attachment(mut self, handle: Handle<FrameGraphTexture>) -> Self {
        self
    }

    pub fn read(mut self, handle: Handle<FrameGraphTexture>, access: vk_sync::AccessType) -> Self {
        self.reads.push((handle, access));
        self
    }

    pub fn write(mut self, handle: Handle<FrameGraphTexture>, access: vk_sync::AccessType) -> Self {
        self.writes.push((handle, access));
        self
    }

    pub fn execute(mut self, callback: impl FnOnce(&ash::Device, vk::CommandBuffer, PassResources<'_>) + 'a) -> Self {
        self.callback = Some(Box::new(callback));
        self
    }
}

fn access_to_sync2(
    access: vk_sync::AccessType,
) -> (vk::PipelineStageFlags2, vk::AccessFlags2, vk::ImageLayout) {
    match access {
        vk_sync::AccessType::Nothing => (
            vk::PipelineStageFlags2::NONE,
            vk::AccessFlags2::NONE,
            vk::ImageLayout::UNDEFINED,
        ),
        vk_sync::AccessType::ColorAttachmentWrite => (
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        ),
        vk_sync::AccessType::ColorAttachmentReadWrite => (
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        ),
        vk_sync::AccessType::DepthStencilAttachmentWrite => (
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        ),
        vk_sync::AccessType::DepthStencilAttachmentRead => (
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
            vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        ),
        vk_sync::AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer => (
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_READ,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        ),
        vk_sync::AccessType::FragmentShaderReadDepthStencilInputAttachment => (
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::INPUT_ATTACHMENT_READ,
            vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        ),
        vk_sync::AccessType::FragmentShaderReadColorInputAttachment => (
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::INPUT_ATTACHMENT_READ,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        ),
        vk_sync::AccessType::FragmentShaderWrite => (
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_WRITE,
            vk::ImageLayout::GENERAL,
        ),
        vk_sync::AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer => (
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_READ,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        ),
        vk_sync::AccessType::ComputeShaderWrite => (
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_WRITE,
            vk::ImageLayout::GENERAL,
        ),
        vk_sync::AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer => (
            vk::PipelineStageFlags2::ALL_GRAPHICS | vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_READ,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        ),
        vk_sync::AccessType::AnyShaderWrite => (
            vk::PipelineStageFlags2::ALL_GRAPHICS | vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_WRITE,
            vk::ImageLayout::GENERAL,
        ),
        vk_sync::AccessType::TransferWrite => (
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        ),
        vk_sync::AccessType::TransferRead => (
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_READ,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        ),
        vk_sync::AccessType::Present => (
            vk::PipelineStageFlags2::NONE,
            vk::AccessFlags2::NONE,
            vk::ImageLayout::PRESENT_SRC_KHR,
        ),
        vk_sync::AccessType::General => (
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
            vk::ImageLayout::GENERAL,
        ),
        _ => unimplemented!("access type not mapped: {:?}", access),
    }
}

mod tests {
    use super::*;

    #[test]
    fn test_topological_sort() {
        let dependencies = vec![
            vec![],      
        ];

        let sorted = FrameScope::topological_sort(&dependencies);
        assert_eq!(sorted, vec![0]);
    }
}