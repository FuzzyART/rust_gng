use crate::ecs::manager;

#[derive(Debug)]
pub enum State {
    Init,
    StartNewIteration,
    TrainingCompleted,
    NormalIteration,
    EpochCompleted,
    IterationCompleted,
}

#[derive(Debug)]
struct System {
    pub train_initiated: bool,
    pub dataset_initiated: bool,
    pub train_completed: bool,
    pub iteration_completed: bool,
    pub reshuffle_required: bool,
    pub normal_iteration: bool,
    pub create_neuron_scheduled: bool,
    pub last_sample_reached: bool,
    pub debug_mode: bool,

    pub curr_iteration: usize,
    pub curr_epoch: usize,

    pub sample_order: Vec<usize>,
    pub sample_order_position: usize,
    pub curr_sample_pos: usize,

    pub curr_neuron: usize,

    pub winner_neuron: usize,
    pub second_neuron: usize,

    pub neighbor_neurons: Vec<usize>,
    pub neighbor_neuron_vec_winner: Vec<usize>,
    pub neighbor_neuron_winner: usize,

    pub neighbor_neuron_vec_max_err: Vec<usize>,
    pub neighbor_neuron_max_err: usize,

    pub neuron_max_err: usize,

    pub newest_neuron_id: usize,
    //pub b
}
impl System {
    pub fn init() -> Self {
        Self {
            train_initiated: false,
            dataset_initiated: false,
            train_completed: false,
            iteration_completed: false,
            reshuffle_required: false,
            normal_iteration: false,
            create_neuron_scheduled: false,
            last_sample_reached: false,
            debug_mode: false,

            curr_iteration: 0,
            curr_epoch: 0,
            curr_neuron: 0,

            sample_order: Vec::new(),
            sample_order_position: 0,
            curr_sample_pos: 0,

            neighbor_neurons: Vec::new(),
            neighbor_neuron_vec_winner: Vec::new(),
            neighbor_neuron_winner: 0,
            neighbor_neuron_vec_max_err: Vec::new(),

            neuron_max_err: 0,
            neighbor_neuron_max_err: 0,

            winner_neuron: 0,
            second_neuron: 0,
            newest_neuron_id: 0,
        }
    }
}
pub struct SystemHandler {
    system_man: manager::EntityManager<System>,
}
impl SystemHandler {
    pub fn init() -> Self {
        Self {
            system_man: manager::EntityManager::new(),
        }
    }
    pub fn create_system(&mut self) {
        self.system_man.create(System::init());
    }
    //----------------------------------------------------------
    pub fn set_sample_order_position(&mut self, val: usize) {
        if let Some(obj) = self.system_man.get_mut(0) {
            obj.sample_order_position = val;
        }
    }
    pub fn get_sample_order_position(&self) -> &usize {
        self.system_man
            .get(0)
            .map(|val| &val.sample_order_position)
            .expect("system flag failure")
    }
    //----------------------------------------------------------

    pub fn get_train_initiated(&self) -> bool {
        self.system_man
            .get(0)
            .map(|val| val.train_initiated)
            .expect("system flag failure")
    }

    pub fn get_dataset_initiated(&self) -> bool {
        self.system_man
            .get(0)
            .map(|val| val.dataset_initiated)
            .expect("system flag failure")
    }

    //--------------------------------------------------------------------------------------------------
    pub fn get_train_completed(&self) -> bool {
        self.system_man
            .get(0)
            .map(|val| val.train_completed)
            .expect("system flag failure")
    }
    pub fn set_train_completed(&mut self, val: bool) {
        if let Some(obj) = self.system_man.get_mut(0) {
            obj.train_completed = val;
        }
    }
    //--------------------------------------------------------------------------------------------------
    pub fn get_iteration_completed(&self) -> bool {
        self.system_man
            .get(0)
            .map(|val| val.iteration_completed)
            .expect("system flag failure")
    }
    pub fn set_iteration_completed(&mut self, val: bool) {
        if let Some(obj) = self.system_man.get_mut(0) {
            obj.iteration_completed = val;
        }
    }
    //--------------------------------------------------------------------------------------------------
    pub fn get_reshuffle_required(&self) -> bool {
        self.system_man
            .get(0)
            .map(|val| val.reshuffle_required)
            .expect("system flag failure")
    }

    pub fn get_normal_iteration(&self) -> bool {
        self.system_man
            .get(0)
            .map(|val| val.normal_iteration)
            .expect("system flag failure")
    }

    //----------------------------------------
    pub fn get_create_neuron_scheduled(&self) -> bool {
        self.system_man
            .get(0)
            .map(|val| val.create_neuron_scheduled)
            .expect("system flag failure")
    }
    pub fn set_create_neuron_scheduled(&mut self, val: bool) {
        if let Some(obj) = self.system_man.get_mut(0) {
            obj.create_neuron_scheduled = val;
        }
    }
    //----------------------------------------
    pub fn get_last_sample_reached(&self) -> bool {
        self.system_man
            .get(0)
            .map(|val| val.last_sample_reached)
            .expect("system flag failure")
    }
    pub fn set_last_sample_reached(&mut self, val: bool) {
        if let Some(obj) = self.system_man.get_mut(0) {
            obj.last_sample_reached = val;
        }
    }

    pub fn get_debug_mode(&self) -> bool {
        self.system_man
            .get(0)
            .map(|val| val.debug_mode)
            .expect("system flag failure")
    }

    pub fn get_curr_iteration(&self) -> usize {
        self.system_man
            .get(0)
            .map(|val| val.curr_epoch)
            .expect("system flag failure")
    }
    pub fn inc_curr_iteration(&mut self) {
        if let Some(obj) = self.system_man.get_mut(0) {
            obj.curr_epoch += 1;
        }
    }
    pub fn get_curr_epoch(&self) -> usize {
        self.system_man
            .get(0)
            .map(|val| val.curr_epoch)
            .expect("system flag failure")
    }
    pub fn set_curr_epoch(&mut self, val: usize) {
        if let Some(obj) = self.system_man.get_mut(0) {
            obj.curr_epoch = val;
        }
    }

    pub fn get_curr_sample_pos(&self) -> usize {
        self.system_man
            .get(0)
            .map(|val| val.curr_sample_pos)
            .expect("system flag failure")
    }
    pub fn set_curr_sample_pos(&mut self, val: usize) {
        if let Some(obj) = self.system_man.get_mut(0) {
            obj.curr_sample_pos = val;
        }
    }
    pub fn get_curr_neuron(&self) -> usize {
        self.system_man
            .get(0)
            .map(|val| val.curr_neuron)
            .expect("system flag failure")
    }
    pub fn set_curr_neuron(&mut self, val: usize) {
        if let Some(obj) = self.system_man.get_mut(0) {
            obj.curr_neuron = val;
        }
    }
    pub fn set_neighbor_neuron_max_err(&mut self, val: usize) {
        if let Some(obj) = self.system_man.get_mut(0) {
            obj.neighbor_neuron_max_err = val;
        }
    }
    pub fn get_neighbor_neuron_max_err(&self) -> &usize {
        self.system_man
            .get(0)
            .map(|val| &val.neighbor_neuron_max_err)
            .expect("system flag failure")
    }

    pub fn set_neuron_max_err(&mut self, val: usize) {
        if let Some(obj) = self.system_man.get_mut(0) {
            obj.neuron_max_err = val;
        }
    }
    pub fn get_neuron_max_err(&self) -> &usize {
        self.system_man
            .get(0)
            .map(|val| &val.neuron_max_err)
            .expect("system flag failure")
    }

    pub fn set_winner_neuron(&mut self, val: usize) {
        if let Some(obj) = self.system_man.get_mut(0) {
            obj.winner_neuron = val;
        }
    }
    pub fn get_winner_neuron(&self) -> &usize {
        self.system_man
            .get(0)
            .map(|val| &val.winner_neuron)
            .expect("system flag failure")
    }

    pub fn set_second_neuron(&mut self, val: usize) {
        if let Some(obj) = self.system_man.get_mut(0) {
            obj.second_neuron = val;
        }
    }
    pub fn get_second_neuron(&self) -> &usize {
        self.system_man
            .get(0)
            .map(|val| &val.second_neuron)
            .expect("system flag failure")
    }

    pub fn set_newest_neuron_id(&mut self, val: usize) {
        if let Some(obj) = self.system_man.get_mut(0) {
            obj.newest_neuron_id = val;
        }
    }
    pub fn get_newest_neuron_id(&self) -> &usize {
        self.system_man
            .get(0)
            .map(|val| &val.newest_neuron_id)
            .expect("system flag failure")
    }

    pub fn set_sample_order(&mut self, val: Vec<usize>) {
        if let Some(obj) = self.system_man.get_mut(0) {
            obj.sample_order = val;
        }
    }
    pub fn get_sample_order(&self) -> &Vec<usize> {
        self.system_man
            .get(0)
            .map(|val| &val.sample_order)
            .expect("system flag failure")
    }

    pub fn set_neighbor_neurons(&mut self, val: Vec<usize>) {
        if let Some(obj) = self.system_man.get_mut(0) {
            obj.neighbor_neurons = val;
        }
    }
    pub fn get_neighbor_neurons(&self) -> &Vec<usize> {
        self.system_man
            .get(0)
            .map(|val| &val.neighbor_neurons)
            .expect("system flag failure")
    }

    pub fn set_neighbor_neuron_vec_winner(&mut self, val: Vec<usize>) {
        if let Some(obj) = self.system_man.get_mut(0) {
            obj.neighbor_neuron_vec_winner = val;
        }
    }
    pub fn get_neighbor_neuron_vec_winner(&self) -> &Vec<usize> {
        self.system_man
            .get(0)
            .map(|val| &val.neighbor_neuron_vec_winner)
            .expect("system flag failure")
    }

    pub fn set_neighbor_neuron_winner(&mut self, val: usize) {
        if let Some(obj) = self.system_man.get_mut(0) {
            obj.neighbor_neuron_winner = val;
        }
    }
    pub fn get_neighbor_neuron_winner(&self) -> &usize {
        self.system_man
            .get(0)
            .map(|val| &val.neighbor_neuron_winner)
            .expect("system flag failure")
    }

    pub fn set_neighbor_neuron_vec_max_err(&mut self, val: Vec<usize>) {
        if let Some(obj) = self.system_man.get_mut(0) {
            obj.neighbor_neuron_vec_max_err = val;
        }
    }
    pub fn get_neighbor_neuron_vec_max_err(&self) -> &Vec<usize> {
        self.system_man
            .get(0)
            .map(|val| &val.neighbor_neuron_vec_max_err)
            .expect("system flag failure")
    }

    // Additional setter methods for other fields
    pub fn set_train_initiated(&mut self, val: bool) {
        if let Some(obj) = self.system_man.get_mut(0) {
            obj.train_initiated = val;
        }
    }

    pub fn set_dataset_initiated(&mut self, val: bool) {
        if let Some(obj) = self.system_man.get_mut(0) {
            obj.dataset_initiated = val;
        }
    }

    pub fn set_reshuffle_required(&mut self, val: bool) {
        if let Some(obj) = self.system_man.get_mut(0) {
            obj.reshuffle_required = val;
        }
    }

    pub fn set_normal_iteration(&mut self, val: bool) {
        if let Some(obj) = self.system_man.get_mut(0) {
            obj.normal_iteration = val;
        }
    }

    pub fn set_debug_mode(&mut self, val: bool) {
        if let Some(obj) = self.system_man.get_mut(0) {
            obj.debug_mode = val;
        }
    }
}
