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
}
