//! A module for managing entities with unique identifiers.
//!
//! The [`EntityManager`] provides a way to create, retrieve, update, and remove entities
//! while efficiently reusing identifiers. It uses a hash map for fast lookups and a deque
//! to track free identifiers for reuse.

use std::collections::{HashMap, VecDeque};

//============================================================
/// A manager for entities that assigns unique IDs and allows efficient retrieval.
///
/// The `EntityManager` stores entities in a [`HashMap`] indexed by `usize` IDs. It
/// maintains a list of reusable IDs in a [`VecDeque`] to minimize ID allocation overhead.
///
/// # Examples
///
/// ```ignore
/// use gng::ecs::manager::EntityManager;
///
/// let mut manager = EntityManager::<String>::new();
/// let id1 = manager.create("entity1".to_string());
/// let id2 = manager.create("entity2".to_string());
///
/// assert_eq!(manager.get(id1), Some(&"entity1".to_string()));
/// assert_eq!(manager.remove(id1), Some("entity1".to_string()));
/// assert_eq!(manager.get(id1), None);
/// ```
#[derive(Debug)]
pub struct EntityManager<T> {
    /// Map of entity IDs to entities.
    entities: HashMap<usize, T>,
    /// Queue of previously used IDs that can be reused.
    free_ids: VecDeque<usize>,
    /// Next ID to assign if no free IDs are available.
    next_id: usize,
}

impl<T> EntityManager<T> {
    /// Creates a new, empty `EntityManager`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use gng::ecs::manager::EntityManager;
    ///
    /// let manager = EntityManager::<String>::new();
    /// assert_eq!(manager.len(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            free_ids: VecDeque::new(),
            next_id: 0,
        }
    }

    /// Creates a new entity with the given value and returns its unique ID.
    ///
    /// If there are previously removed IDs that can be reused, this function will
    /// assign one of them. Otherwise, it assigns the next sequential ID.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to store as an entity.
    ///
    /// # Returns
    ///
    /// Returns the unique ID assigned to the new entity.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use gng::ecs::manager::EntityManager;
    ///
    /// let mut manager = EntityManager::<String>::new();
    /// let id = manager.create("hello".to_string());
    /// assert_eq!(id, 0);
    /// ```
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

    /// Removes the entity with the given ID and returns it if it exists.
    ///
    /// The ID of the removed entity is added to the list of free IDs for reuse.
    ///
    /// # Arguments
    ///
    /// * `key` - The unique ID of the entity to remove.
    ///
    /// # Returns
    ///
    /// Returns `Some(T)` containing the removed entity if it existed, or `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use gng::ecs::manager::EntityManager;
    ///
    /// let mut manager = EntityManager::<String>::new();
    /// let id = manager.create("hello".to_string());
    /// let entity = manager.remove(id);
    /// assert_eq!(entity, Some("hello".to_string()));
    /// ```
    pub fn remove(&mut self, key: usize) -> Option<T> {
        if let Some(value) = self.entities.remove(&key) {
            self.free_ids.push_back(key);
            Some(value)
        } else {
            None
        }
    }

    /// Gets a reference to the entity with the given ID.
    ///
    /// # Arguments
    ///
    /// * `key` - The unique ID of the entity to retrieve.
    ///
    /// # Returns
    ///
    /// Returns `Some(&T)` if an entity with the given ID exists, or `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use gng::ecs::manager::EntityManager;
    ///
    /// let mut manager = EntityManager::<String>::new();
    /// let id = manager.create("hello".to_string());
    /// assert_eq!(manager.get(id), Some(&"hello".to_string()));
    /// ```
    pub fn get(&self, key: usize) -> Option<&T> {
        self.entities.get(&key)
    }

    /// Gets a mutable reference to the entity with the given ID.
    ///
    /// # Arguments
    ///
    /// * `key` - The unique ID of the entity to retrieve.
    ///
    /// # Returns
    ///
    /// Returns `Some(&mut T)` if an entity with the given ID exists, or `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use gng::ecs::manager::EntityManager;
    ///
    /// let mut manager = EntityManager::<String>::new();
    /// let id = manager.create("hello".to_string());
    /// if let Some(entity) = manager.get_mut(id) {
    ///     *entity = "world".to_string();
    /// }
    /// assert_eq!(manager.get(id), Some(&"world".to_string()));
    /// ```
    pub fn get_mut(&mut self, key: usize) -> Option<&mut T> {
        self.entities.get_mut(&key)
    }

    /// Returns an iterator over all entities.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use gng::ecs::manager::EntityManager;
    ///
    /// let mut manager = EntityManager::<String>::new();
    /// let _id1 = manager.create("hello".to_string());
    /// let _id2 = manager.create("world".to_string());
    ///
    /// for (id, entity) in manager.iter() {
    ///     println!("ID: {}, Entity: {}", id, entity);
    /// }
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = (&usize, &T)> {
        self.entities.iter()
    }

    /// Returns a mutable iterator over all entities.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use gng::ecs::manager::EntityManager;
    ///
    /// let mut manager = EntityManager::<String>::new();
    /// let _id1 = manager.create("hello".to_string());
    /// let _id2 = manager.create("world".to_string());
    ///
    /// for (_id, entity) in manager.iter_mut() {
    ///     *entity = entity.to_uppercase();
    /// }
    /// ```
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&usize, &mut T)> {
        self.entities.iter_mut()
    }

    /// Returns the number of entities currently stored.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use gng::ecs::manager::EntityManager;
    ///
    /// let mut manager = EntityManager::<String>::new();
    /// assert_eq!(manager.len(), 0);
    /// let _id = manager.create("hello".to_string());
    /// assert_eq!(manager.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// Returns an iterator over the keys (IDs) of all entities.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use gng::ecs::manager::EntityManager;
    ///
    /// let mut manager = EntityManager::<String>::new();
    /// let id1 = manager.create("hello".to_string());
    /// let id2 = manager.create("world".to_string());
    ///
    /// let ids: Vec<usize> = manager.keys().cloned().collect();
    /// assert_eq!(ids, vec![id1, id2]);
    /// ```
    pub fn keys(&self) -> impl Iterator<Item = &usize> {
        self.entities.keys()
    }
}
