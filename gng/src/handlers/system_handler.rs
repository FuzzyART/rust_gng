use crate::ecs::manager;

#[derive(Debug, PartialEq, Clone)]
pub enum Phase {
    StartNewEpoch,
    NormalIteration,
    IterationCompleted,
}

#[derive(Debug)]
pub struct System {
    pub curr_phase: Phase,
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
}
impl System {
    pub fn init() -> Self {
        Self {
            curr_phase: Phase::StartNewEpoch,
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

    // Getters and Setters
    //pub fn get_curr_phase(&self)->Phase{
    //    self.curr_phase.clone()
    //}
    //pub fn set_curr_phase(&mut self,p: Phase){
    //    self.curr_phase = p;
    //}

    //pub fn get_train_initiated(&self) -> bool {
    //    self.train_initiated
    //}

    //pub fn set_train_initiated(&mut self, val: bool) {
    //    self.train_initiated = val;
    //}

    //pub fn get_dataset_initiated(&self) -> bool {
    //    self.dataset_initiated
    //}

    //pub fn set_dataset_initiated(&mut self, val: bool) {
    //    self.dataset_initiated = val;
    //}

    //pub fn get_train_completed(&self) -> bool {
    //    self.train_completed
    //}

    //pub fn set_train_completed(&mut self, val: bool) {
    //    self.train_completed = val;
    //}

    //pub fn get_iteration_completed(&self) -> bool {
    //    self.iteration_completed
    //}

    //pub fn set_iteration_completed(&mut self, val: bool) {
    //    self.iteration_completed = val;
    //}

    //pub fn get_reshuffle_required(&self) -> bool {
    //    self.reshuffle_required
    //}

    //pub fn set_reshuffle_required(&mut self, val: bool) {
    //    self.reshuffle_required = val;
    //}

    //pub fn get_normal_iteration(&self) -> bool {
    //    self.normal_iteration
    //}

    //pub fn set_normal_iteration(&mut self, val: bool) {
    //    self.normal_iteration = val;
    //}

    //pub fn get_create_neuron_scheduled(&self) -> bool {
    //    self.create_neuron_scheduled
    //}

    //pub fn set_create_neuron_scheduled(&mut self, val: bool) {
    //    self.create_neuron_scheduled = val;
    //}

    //pub fn get_last_sample_reached(&self) -> bool {
    //    self.last_sample_reached
    //}

    //pub fn set_last_sample_reached(&mut self, val: bool) {
    //    self.last_sample_reached = val;
    //}

    //pub fn get_debug_mode(&self) -> bool {
    //    self.debug_mode
    //}

    //pub fn set_debug_mode(&mut self, val: bool) {
    //    self.debug_mode = val;
    //}

    //pub fn get_curr_iteration(&self) -> usize {
    //    self.curr_iteration
    //}

    //pub fn set_curr_iteration(&mut self, val: usize) {
    //    self.curr_iteration = val;
    //}

    //pub fn inc_curr_iteration(&mut self) {
    //    self.curr_iteration += 1;
    //}

    //pub fn get_curr_epoch(&self) -> usize {
    //    self.curr_epoch
    //}

    //pub fn set_curr_epoch(&mut self, val: usize) {
    //    self.curr_epoch = val;
    //}

    //pub fn get_curr_sample_pos(&self) -> usize {
    //    self.curr_sample_pos
    //}

    //pub fn set_curr_sample_pos(&mut self, val: usize) {
    //    self.curr_sample_pos = val;
    //}

    //pub fn get_curr_neuron(&self) -> usize {
    //    self.curr_neuron
    //}

    //pub fn set_curr_neuron(&mut self, val: usize) {
    //    self.curr_neuron = val;
    //}

    //   pub fn get_neighbor_neuron_max_err(&self) -> usize {
    //       self.neighbor_neuron_max_err
    //   }

    //pub fn set_neighbor_neuron_max_err(&mut self, val: usize) {
    //    self.neighbor_neuron_max_err = val;
    //}

    //pub fn get_neuron_max_err(&self) -> usize {
    //    self.neuron_max_err
    //}

    //pub fn set_neuron_max_err(&mut self, val: usize) {
    //    self.neuron_max_err = val;
    //}

    //pub fn get_winner_neuron(&self) -> usize {
    //    self.winner_neuron
    //}

    //pub fn set_winner_neuron(&mut self, val: usize) {
    //    self.winner_neuron = val;
    //}

    //pub fn get_second_neuron(&self) -> usize {
    //    self.second_neuron
    //}

    //pub fn set_second_neuron(&mut self, val: usize) {
    //    self.second_neuron = val;
    //}

    //pub fn get_newest_neuron_id(&self) -> usize {
    //    self.newest_neuron_id
    //}

    //pub fn set_newest_neuron_id(&mut self, val: usize) {
    //    self.newest_neuron_id = val;
    //}

    //pub fn get_sample_order(&self) -> &Vec<usize> {
    //    &self.sample_order
    //}

    //pub fn set_sample_order(&mut self, val: Vec<usize>) {
    //    self.sample_order = val;
    //}

    //pub fn get_sample_order_position(&self) -> usize {
    //    self.sample_order_position
    //}

    //pub fn set_sample_order_position(&mut self, val: usize) {
    //    self.sample_order_position = val;
    //}

    //pub fn get_neighbor_neurons(&self) -> &Vec<usize> {
    //    &self.neighbor_neurons
    //}

    //pub fn set_neighbor_neurons(&mut self, val: Vec<usize>) {
    //    self.neighbor_neurons = val;
    //}

    //pub fn get_neighbor_neuron_vec_winner(&self) -> &Vec<usize> {
    //    &self.neighbor_neuron_vec_winner
    //}

    //pub fn set_neighbor_neuron_vec_winner(&mut self, val: Vec<usize>) {
    //    self.neighbor_neuron_vec_winner = val;
    //}

    //pub fn get_neighbor_neuron_winner(&self) -> usize {
    //    self.neighbor_neuron_winner
    //}

    //pub fn set_neighbor_neuron_winner(&mut self, val: usize) {
    //    self.neighbor_neuron_winner = val;
    //}

    //pub fn get_neighbor_neuron_vec_max_err(&self) -> &Vec<usize> {
    //    &self.neighbor_neuron_vec_max_err
    //}

    //pub fn set_neighbor_neuron_vec_max_err(&mut self, val: Vec<usize>) {
    //    self.neighbor_neuron_vec_max_err = val;
    //}
}
