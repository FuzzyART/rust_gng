use crate::ecs::manager;
//--------------------------------------------------------------------------------------------------
#[derive(Debug)]
pub struct Edge {
    pub start: usize,
    pub end: usize,
    pub age: usize,
}
impl Edge {
    pub fn init(from: usize, to: usize, age: usize) -> Self {
        Self {
            start: from,
            end: to,
            age: age,
        }
    }
}

//--------------------------------------------------------------------------------------------------

pub struct EdgeHandler {
    pub edge_man: manager::EntityManager<Edge>,
}
impl EdgeHandler {
    pub fn init() -> Self {
        Self {
            edge_man: manager::EntityManager::new(),
        }
    }
    pub fn create_edge(&mut self, from: usize, to: usize, age: usize) {
        self.edge_man.create(Edge::init(from, to, age));
    }
    pub fn remove_edge(&mut self, key: usize) {
        self.edge_man.remove(key);
    }

    pub fn print_edges(&mut self) {
        for (id, val) in self.edge_man.iter() {
            println!("ID: {} From: {:?} To: {:?}", id, val.start, val.end);
        }
    }

    pub fn get_connected_edges(&self, neuron_num: usize) -> Vec<usize> {
        let mut res: Vec<usize> = Vec::new();
        let keys = self.get_keys();
        for a in keys {
            let start_edge: usize = *self
                .edge_man
                .get(*a)
                .map(|val| &val.start)
                .expect("edge not found 1");
            let end_edge: usize = *self
                .edge_man
                .get(*a)
                .map(|val| &val.end)
                .expect("edge not found 2");
            if (start_edge == neuron_num) || (end_edge == neuron_num) {
                res.push(*a);
            }
        }
        res
    }
    pub fn get_edge_start(&self, num_neuron: usize) -> &usize {
        let edge_start = self
            .edge_man
            .get(num_neuron)
            .map(|val| &val.start)
            .expect("edge start not found 1");
        edge_start
    }
    pub fn get_edge_end(&self, num: usize) -> &usize {
        let edge_start = self
            .edge_man
            .get(num)
            .map(|val| &val.end)
            .expect("edge start not found 2");
        edge_start
    }
    pub fn get_edge_age(&self, num: usize) -> &usize {
        let edge_age = self
            .edge_man
            .get(num)
            .map(|val| &val.age)
            .expect("edge age not found 3");
        edge_age
    }
    pub fn set_edge_age(&mut self, num: usize, val: usize) {
        if let Some(obj) = self.edge_man.get_mut(num) {
            obj.age = val;
        }
    }
    pub fn increase_edge_age(&mut self, num: usize) {
        if let Some(obj) = self.edge_man.get_mut(num) {
            obj.age = obj.age + 1;
        }
    }
    pub fn len(&self) -> usize {
        self.edge_man.len()
    }
    //==================================================================================================
    // Debug functions
    pub fn get_edge_start_vec(&self) -> Vec<usize> {
        let len = self.edge_man.len();
        let mut res_vec: Vec<usize> = vec![0; len];
        //for key in self.edge_man.keys(){
        //if let Some(entity) = self.edge_man.get(*key){

        for (id, line) in self.edge_man.iter() {
            res_vec[*id] = line.start;
        }
        res_vec
    }
    pub fn get_edge_end_vec(&self) -> Vec<usize> {
        let len = self.edge_man.len();
        let mut res_vec: Vec<usize> = vec![0; len];
        //for key in self.edge_man.keys(){
        //if let Some(entity) = self.edge_man.get(*key){

        for (id, line) in self.edge_man.iter() {
            res_vec[*id] = line.end;
        }
        res_vec
    }
    pub fn get_edge_age_vec(&self) -> Vec<usize> {
        let len = self.edge_man.len();
        let mut res_vec: Vec<usize> = vec![0; len];

        for (id, line) in self.edge_man.iter() {
            res_vec[*id] = line.age;
        }
        res_vec
    }
    pub fn get_all_edge_ids(&self) -> Vec<usize> {
        self.edge_man.keys().copied().collect()
    }
    pub fn get_keys(&self) -> impl Iterator<Item = &usize> {
        self.edge_man.keys()
    }
}
