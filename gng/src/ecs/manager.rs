use std::collections::{HashMap, VecDeque};

//============================================================
#[derive(Debug)]
pub struct EntityManager<T> {
    entities: HashMap<usize, T>,
    free_ids: VecDeque<usize>,
    next_id: usize,
}
impl<T> EntityManager<T> {
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            free_ids: VecDeque::new(),
            next_id: 0,
        }
    }

    pub fn create(&mut self, value: T) -> usize {
        if let Some(id) = self.free_ids.pop_front() {
            self.entities.insert(id, value);
            id
        } else {
            let id = self.next_id;
            self.next_id += 1;
            self.entities.insert(id, value);
            id
        }
    }

    pub fn remove(&mut self, key: usize) -> Option<T> {
        if let Some(value) = self.entities.remove(&key) {
            self.free_ids.push_back(key);
            Some(value)
        } else {
            None
        }
    }
    pub fn get(&self, key: usize) -> Option<&T> {
        self.entities.get(&key)
    }
    pub fn get_mut(&mut self, key: usize) -> Option<&mut T> {
        self.entities.get_mut(&key)
    }
    pub fn iter(&self) -> impl Iterator<Item = (&usize, &T)> {
        self.entities.iter()
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&usize, &mut T)> {
        self.entities.iter_mut()
    }
    pub fn len(&self) -> usize {
        self.entities.len()
    }
    pub fn keys(&self) -> impl Iterator<Item = &usize> {
        self.entities.keys()
    }
}
