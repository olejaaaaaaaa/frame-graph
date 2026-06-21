use std::marker::PhantomData;
use slotmap::{new_key_type};

new_key_type! {
    pub(crate) struct Key;
}

#[derive(Clone, Copy)]
pub struct Handle<T> {
    pub(crate) key: Key,
    pub(crate) _marker: PhantomData<T>
}

impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
    fn ne(&self, other: &Self) -> bool {
        self.key != other.key
    }
}