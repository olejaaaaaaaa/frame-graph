use slotmap::new_key_type;
use std::marker::PhantomData;

new_key_type! {
    pub(crate) struct Key;
}

pub struct Handle<T> {
    pub(crate) key: Key,
    pub(crate) _marker: PhantomData<T>,
}

impl<T> std::fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Handle")
            .field("key", &self.key)
            .field("_marker", &self._marker)
            .finish()
    }
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Handle {
            key: self.key.clone(),
            _marker: PhantomData,
        }
    }
}

impl<T> Copy for Handle<T> {}

impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
    fn ne(&self, other: &Self) -> bool {
        self.key != other.key
    }
}
