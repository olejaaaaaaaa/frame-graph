#[derive(Debug)]
pub enum FrameGraphError {
    Allocation,
    CreateVkImage,
    BindVkImage,
}

pub type Result<T> = std::result::Result<T, FrameGraphError>;