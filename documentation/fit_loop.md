# GNG Training Loop - fit() Function

## UML Activity Diagram

```mermaid
flowchart TD
    A[Start] --> B{Training Complete?}
    B -- No --> C[Update State]
    C --> D[Get Current State]
    D --> E{State}
    E -->|Init| F[init_training]
    F --> G[shuffle_dataset]
    G --> H[set_train_initiated]
    H --> I[Set State to NormalIteration]
    E -->|NormalIteration| J[select_sample]
    J --> K[calc_neuron_distances]
    K --> L[calc_nearest_neurons]
    L --> M[calc_neuron_dependencies]
    M --> N[increase_edge_age]
    N --> O[add_error_to_winner_neuron]
    O --> P[update_weights]
    P --> Q[create_edge]
    Q --> R[delete_old_edges]
    R --> S[remove_unconnected_neurons]
    S --> T{create_neuron_scheduled?}
    T -- Yes --> U[create_neuron]
    U --> V[set_create_neuron_scheduled]
    T -- No --> W[decrease_error_global]
    W --> X[Increment iteration counter]
    X --> Y[Update State]
    Y --> B
    E -->|EpochCompleted| Z[start_new_epoch]
    Z --> AA[check_stopping_criterion]
    AA --> AB[Set State to NormalIteration]
    E -->|StartNewIteration| AC[shuffle_dataset]
    AC --> AD[Increment epoch]
    AD --> AE[Set iteration_completed]
    AE --> AF[Set State to TrainingCompleted]
    E -->|TrainingCompleted| AG[end_loop]
    AG --> AH[Set Training Complete]
    AH --> B
    E -->|IterationCompleted| AI[Continue Loop]
    AI --> B
```

## Function Overview

The `fit` function implements the main training loop for the Growing Neural Gas (GNG) algorithm. It manages the entire training process through multiple states and iterations until the stopping criterion is met.

## Key Components

1. **State Management**: The function uses a `State` enum to control the flow through different phases of training
2. **Training Loop**: Continuous loop that processes samples and updates network structure
3. **Neural Gas Operations**: Core GNG operations including neuron creation, edge management, and weight updates
4. **Stopping Criteria**: Checks for termination conditions at epoch boundaries

## State Transitions

- **Init**: Initialize training parameters and dataset
- **NormalIteration**: Main training cycle processing samples and updating network
- **EpochCompleted**: Handle epoch completion and stopping criterion checks
- **StartNewIteration**: Prepare for next iteration
- **TrainingCompleted**: Finalize training process

## Core Operations

- Sample selection and processing
- Neuron distance calculations
- Nearest neuron identification
- Edge age management
- Error accumulation and propagation
- Weight updates
- Edge creation and deletion
- Neuron creation when needed
- Global error reduction


```mermaid
flowchart TD
    Start([Start update_state]) --> GetValues[Get Current Epoch & Max Iterations]
    GetValues --> CheckMaxEpoch{curr_epoch >= max_train_iterations?}
    CheckMaxEpoch -- Yes --> SetCompleted[Set train_completed = true]
    SetCompleted --> CheckInitiated{get_train_initiated}
    CheckMaxEpoch -- No --> CheckInitiated
    CheckInitiated -- Yes --> SetNormalIter[Set state = NormalIteration]
    CheckInitiated -- No --> CheckLastSample{get_last_sample_reached}
    SetNormalIter --> CheckLastSample
    CheckLastSample -- Yes --> ResetLastSample[Set last_sample_reached = false]
    ResetLastSample --> SetStartNewIter[Set state = StartNewIteration]
    SetStartNewIter --> CheckCreation{curr_iteration % creation_interval == 0?}
    CheckLastSample -- No --> CheckCreation
    CheckCreation -- Yes --> ScheduleNeuron[Set create_neuron_scheduled = true]
    CheckCreation -- No --> End([End update_state])
    ScheduleNeuron --> End
    SetCompleted --> CheckInitiated
    SetCompleted --> CheckLastSample
    SetCompleted --> CheckCreation
    SetNormalIter --> CheckLastSample
    SetNormalIter --> CheckCreation
    SetStartNewIter --> CheckCreation
    ResetLastSample --> CheckCreation
```



```mermaid
stateDiagram-v2
    [*] --> StartUpdateState
    StartUpdateState --> GetValues: Start update_state
    GetValues --> CheckMaxEpoch: Get Current Epoch & Max Iterations
    
    CheckMaxEpoch --> SetCompleted: curr_epoch >= max_train_iterations?
    CheckMaxEpoch --> CheckInitiated: curr_epoch < max_train_iterations?
    
    SetCompleted --> CheckInitiated: Set train_completed = true
    
    CheckInitiated --> SetNormalIter: get_train_initiated?
    CheckInitiated --> CheckLastSample: get_train_initiated?
    
    SetNormalIter --> CheckLastSample: Set state = NormalIteration
    
    CheckLastSample --> ResetLastSample: last_sample_reached?
    CheckLastSample --> CheckCreation: last_sample_reached?
    
    ResetLastSample --> SetStartNewIter: Set last_sample_reached = false
    
    SetStartNewIter --> CheckCreation: Set state = StartNewIteration
    
    CheckCreation --> ScheduleNeuron: curr_iteration % creation_interval == 0?
    CheckCreation --> End: curr_iteration % creation_interval != 0?
    
    ScheduleNeuron --> End: Set create_neuron_scheduled = true
    
    [*] --> StartUpdateState
    StartUpdateState --> GetValues
    GetValues --> CheckMaxEpoch
    CheckMaxEpoch --> SetCompleted
    CheckMaxEpoch --> CheckInitiated
    SetCompleted --> CheckInitiated
    CheckInitiated --> SetNormalIter
    CheckInitiated --> CheckLastSample
    SetNormalIter --> CheckLastSample
    CheckLastSample --> ResetLastSample
    ResetLastSample --> SetStartNewIter
    SetStartNewIter --> CheckCreation
    CheckLastSample --> CheckCreation
    CheckCreation --> ScheduleNeuron
    ScheduleNeuron --> End
    ResetLastSample --> CheckCreation
```
# FIT + State change
```mermaid
graph TD
    A[Start] --> B[Set current_state = Init]
    B --> C{TrainCompleted?}
    C -- No --> D[Init Training]
    D --> E[Shuffle Dataset]
    E --> F[Set Train Initiated = true]
    F --> G[Set current_state = NormalIteration]
    C -- Yes --> H[Stop Training]
    H --> I[End]
    
    G --> J[Loop Start]
    J --> K[Update State]
    K --> L{current_state}
    
    L -->|MATCH Init| M[Init Training]
    M --> N[Shuffle Dataset]
    N --> O[Set Train Initiated = true]
    O --> P[Set current_state = NormalIteration]
    
    L -->|NormalIteration| Q[Select Sample]
    Q --> R[Calc Neuron Distances]
    R --> S[Calc Nearest Neurons]
    S --> T[Calc Neuron Dependencies]
    T --> U[Increase Edge Age]
    U --> V[Add Error to Winner Neuron]
    V --> W[Update Weights]
    W --> X[Create Edge]
    X --> Y[Delete Old Edges]
    Y --> Z[Remove Unconnected Neurons]
    Z --> AA{Create Neuron Scheduled?}
    AA -- Yes --> AB[Create Neuron]
    AB --> AC[Set Create Neuron Scheduled = false]
    AC --> AD[Decrease Global Error]
    AD --> AE{Last Sample Reached?}
    AE -- Yes --> AF[Set current_state = StartNewIteration]
    AE -- No --> AG[Continue NormalIteration]
    
    L -->|StartNewIteration| AH[Shuffle Dataset]
    AH --> AI[Increment Epoch]
    AI --> AJ[Set Iteration Completed = true]
    AJ --> AK[Set current_state = TrainingCompleted]
    
    L -->|TrainingCompleted| AL[End Loop]
    
    L -->|EpochCompleted| AM[Start New Epoch]
    AM --> AN[Check Stopping Criterion]
    AN --> AO[Set current_state = NormalIteration]
    
    L -->|IterationCompleted| AP[Do Nothing]
    
    AO --> AQ[Increment Current Iteration]
    AQ --> AR[Loop Back to Check TrainCompleted]
    AR --> C
    
    style A fill:#e1f5fe
    style B fill:#fff3e0
    style C fill:#fce4ec
    style G fill:#e8f5e9
    style J fill:#fff3e0
    style K fill:#fce4ec
    style L fill:#e1f5fe
    style M fill:#fff3e0
    style Q fill:#e8f5e9
    style AA fill:#fce4ec
    style AE fill:#fce4ec
    style AH fill:#fff3e0
    style AK fill:#fce4ec
    style AL fill:#e1f5fe
    style AM fill:#fff3e0
    style AO fill:#e8f5e9
    style AQ fill:#fff3e0
    style AR fill:#e1f5fe
```






# NEW

```mermaid
graph TD
    A[update_state Function] --> B[Get current epoch]
    B --> C{curr_epoch >= max_epochs?}
    C -->|Yes| D[Set train_completed = true]
    C -->|No| E[Check train_initiated]
    E --> F{train_initiated?}
    F -->|Yes| G[Set state = NormalIteration]
    F -->|No| H[Continue]
    H --> I[Check last_sample_reached]
    I --> J{last_sample_reached?}
    J -->|Yes| K[Set last_sample_reached = false]
    K --> L[Set state = StartNewIteration]
    J -->|No| M[Check neuron creation interval]
    M --> N{curr_iteration % neuron_creation_interval == 0?}
    N -->|Yes| O[Set create_neuron_scheduled = true]
    N -->|No| P[End]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#fff3e0
    style D fill:#e8f5e9
    style E fill:#f3e5f5
    style F fill:#fff3e0
    style G fill:#e8f5e9
    style H fill:#f3e5f5
    style I fill:#f3e5f5
    style J fill:#fff3e0
    style K fill:#e8f5e9
    style L fill:#e8f5e9
    style M fill:#f3e5f5
    style N fill:#fff3e0
    style O fill:#e8f5e9
    style P fill:#f3e5f5
```
