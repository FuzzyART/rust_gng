pub struct Params {
    pub num_neurons: usize, // variable, to check neuron number against
    pub input_width: usize, // variable, to check input widht against

    //----------------------------------------
    // Neurons
    // vvvv Everything in here must have num_neuron sizes vvvv
    pub w: Vec<f64>, // len must be num_neurons*input_width
    pub distance: Vec<f64>,
    pub neuron_err: Vec<f64>,

    pub distance_order: Vec<usize>,
    pub distance_ranking: Vec<usize>, // ordered distances to curr. input. 0 = winner neuron
    pub neuron_err_ranking: Vec<usize>, // first has biggest error
    pub neuron_dependencies: Vec<usize>, // 1: winner neuron,
    // 2: neighbour neuron
    // 0: not connected to winner neuron

    // ^^^^ Everything in here must have num_neuron sizes ^^^^
    //----------------------------------------
    // EDGES
    // vvvv lenghts must be same vvvv
    pub edge_start: Vec<usize>,
    pub edge_end: Vec<usize>,
    pub edge_age: Vec<usize>,
    // ^^^^ lenghts must be same ^^^^
    pub winner_edges: Vec<usize>,
    //----------------------------------------
}
impl Params {
    pub fn init() -> Self {
        Self {
            num_neurons: 0,
            input_width: 0,
            w: Vec::new(),
            neuron_dependencies: Vec::new(),
            distance: Vec::new(),
            distance_order: Vec::new(),
            distance_ranking: Vec::new(),
            neuron_err: Vec::new(),
            neuron_err_ranking: Vec::new(),

            edge_start: Vec::new(),
            edge_end: Vec::new(),
            edge_age: Vec::new(),
            winner_edges: Vec::new(),
        }
    }
}
