use crate::data_structures::input_set_params;
use crate::data_structures::model_params;
use crate::data_structures::train_params;

pub struct GngParams {
    pub train_params: train_params::TrainParams,
    pub model_params: model_params::Params,
    pub algo_state: train_params::AlgorithmState,
    pub input_set_params: input_set_params::Params,
    pub file_names: train_params::FileNames,
}
impl GngParams {
    pub fn init() -> Self {
        Self {
            train_params: train_params::TrainParams::init(),
            algo_state: train_params::AlgorithmState::init(),
            model_params: model_params::Params::init(),
            input_set_params: input_set_params::Params::init(),
            file_names: train_params::FileNames::init(),
        }
    }
}

pub struct GngParams2 {
    pub train_params: train_params::TrainParams,
    pub model_params: model_params::Params,
    pub algo_state: train_params::AlgorithmState,
    pub input_set_params: input_set_params::Params,
    pub file_names: train_params::FileNames,
}
impl GngParams2 {
    pub fn init() -> Self {
        Self {
            train_params: train_params::TrainParams::init(),
            algo_state: train_params::AlgorithmState::init(),
            model_params: model_params::Params::init(),
            input_set_params: input_set_params::Params::init(),
            file_names: train_params::FileNames::init(),
        }
    }
}
