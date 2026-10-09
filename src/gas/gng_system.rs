use crate::handlers::config_handler::Config;
use crate::handlers::system_handler::System;
use crate::handlers::system_handler::Phase;
use crate::components::{
    edge_component::EdgeHandler, neuron_component::NeuronHandler, sample_component::SampleHandler,
};

use crate::aux::{
    json_reader, json_writer::write_json_to_file, json_writer::write_value_to_block,
    rng_manager::RngManager,
};

pub struct Gng_System {
    pub neuron_handler: NeuronHandler,
    pub edge_handler: EdgeHandler,

    pub config_handler: Config,
    pub system_handler: System,
    pub sample_handler: SampleHandler,
    pub rng_manager: RngManager,
}
impl Gng_System {
    pub fn init() -> Self {
        Self {
            neuron_handler: NeuronHandler::init(),
            edge_handler: EdgeHandler::init(),

            config_handler: Config::init(),
            system_handler: System::init(),
            sample_handler: SampleHandler::init(),
            rng_manager: RngManager::init(123),
        }
    }
    pub fn create_system(&mut self) {
        Self::init();
        self.rng_manager.seed_rng(1234567890);
    }
}
