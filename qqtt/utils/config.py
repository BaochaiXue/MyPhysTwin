# Stage 0—Configuration Backbone
# Role: Provide a singleton hyperparameter store shared across trainers, renderers,
#       and dataset loaders; exposes YAML ingestion and runtime overrides.
# Inputs: YAML config files (e.g., `configs/cloth.yaml`), pickled optimisation dicts
#         (`optimal_params.pkl`), imperative overrides via `update_from_dict`.
# Outputs: Mutates module-level `cfg` instance exposing attributes used across the
#          simulation stack; supplies metadata to logging/tracking backends.
# Key in-house deps: `.misc.singleton` decorator, `qqtt.utils` consumers, offline
#                    optimisation pipeline feeding parameters back.
# Side effects: Reads YAML files from disk, updates global mutable state (`cfg`).
# Assumptions: YAML keys align with existing attributes; consumers honour attribute
#              types (ints/floats/bools) and treat `cfg` as globally shared state.

from .misc import singleton
import yaml


@singleton
class Config:
    """Singleton encapsulating simulator, optimisation, and rendering hyperparameters.

    This class is instantiated exactly once by virtue of the `@singleton` decorator,
    ensuring every module in the pipeline reads/writes the same configuration state.
    Attributes are initialised with empirically tuned defaults but can be overridden via
    YAML config files or optimal-parameter pickles produced by offline optimisation.
    """

    def __init__(self):
        """Populate default values that cover both real-world and synthetic workloads."""

        # Data modality; toggles data loader branches (`real` vs `synthetic`).
        self.data_type = "real"
        # Frame rate assumed when converting simulator timesteps to wall-clock seconds.
        self.FPS = 30
        # Base time-step (seconds) used inside Warp kernels; tiny to maintain stability.
        self.dt = 5e-5
        # Number of substeps per rendered frame; ensures CFL-like stability conditions.
        self.num_substeps = round(1.0 / self.FPS / self.dt)

        # Viscous damper attached to springs to dissipate energy.
        self.dashpot_damping = 100
        # Aerodynamic drag coefficient modelling ambient damping on nodes.
        self.drag_damping = 3
        # Learning rate for Adam optimiser during inverse identification.
        self.base_lr = 1e-3
        # Total optimisation iterations after which training stops.
        self.iterations = 250
        # Frequency (in iterations) to trigger visualisation renders.
        self.vis_interval = 10
        # Initial Young's modulus value used to seed simulator material stiffness.
        self.init_spring_Y = 3e3
        # Ground-plane collision elasticity coefficient (bounciness).
        self.collide_elas = 0.5
        # Tangential friction coefficient for ground-plane contact.
        self.collide_fric = 0.3
        # Elasticity for object-object collisions (e.g., controllers vs cloth).
        self.collide_object_elas = 0.7
        # Friction coefficient for controller/object interactions.
        self.collide_object_fric = 0.3

        # Radius threshold for connecting object-object springs in the initial graph.
        self.object_radius = 0.02
        # Upper bound on neighbours to avoid dense, unstable spring networks.
        self.object_max_neighbours = 30
        # Search radius for linking controller points to mesh vertices.
        self.controller_radius = 0.04
        # Limit on controller-to-vertex spring connections to cap constraint stiffness.
        self.controller_max_neighbours = 50

        # Bounds that clamp learnt Young's modulus values during optimisation.
        self.spring_Y_min = 0
        self.spring_Y_max = 1e5

        # Flip Z-axis to match the dataset's coordinate convention (camera looking in -Z).
        self.reverse_z = True
        # Default camera viewing parameters applied inside the visualiser.
        self.vp_front = [1, 0, -2]
        self.vp_up = [0, 0, -1]
        self.vp_zoom = 1

        # Minimum separation distance enforced for collision detection heuristics.
        self.collision_dist = 0.06
        # Toggle to allow gradient updates on collision coefficients during training.
        self.collision_learn = True
        # Enable or disable self-collision handling inside the simulator kernels.
        self.self_collision = False

        # Controls whether Warp's captured graphs accelerate repeated kernel execution.
        self.use_graph = True

        # Weights applied to reconstruction losses when training on real captures.
        self.chamfer_weight = 1.0
        self.track_weight = 1.0
        self.acc_weight = 0.01

        # Optional overlay image path blended into rendered frames for debugging.
        self.overlay_path = None

    def to_dict(self):
        """Return a dictionary snapshot of the configuration state.

        Returns
        -------
        Dict[str, Any]
            Mapping from attribute names to their current values, excluding callables
            and private members; intended for experiment logging (e.g., WandB).
        """

        # Iterate through attributes, skipping callables/private names, so the trainer
        # can serialise the configuration state for experiment tracking (e.g., WandB).
        return {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        }

    def update_from_dict(self, config_dict):
        """Apply key-value overrides while preserving original attribute types.

        Parameters
        ----------
        config_dict : Mapping[str, Any]
            Arbitrary dictionary whose keys correspond to existing attributes. Values
            are cast to the attribute's native type to avoid implicit string parsing.
        """

        for key, value in config_dict.items():
            if hasattr(self, key):
                current_value = getattr(self, key)
                # Python's YAML loader may return strings; cast to match our defaults so
                # downstream code relying on numeric types continues to work correctly.
                if isinstance(current_value, int):
                    value = int(value)
                elif isinstance(current_value, float):
                    value = float(value)
                setattr(self, key, value)

    def load_from_yaml(self, file_path):
        """Load configuration overrides from a YAML file on disk.

        Parameters
        ----------
        file_path : str or pathlib.Path
            Path to a YAML document whose top-level keys mirror Config attribute names.

        Notes
        -----
        * Uses `yaml.safe_load`, which constructs Python primitives (dict/list/str/num)
          without executing arbitrary code, guarding against configuration injection.
        """

        with open(file_path, "r") as file:
            config_dict = yaml.safe_load(file)
        self.update_from_dict(config_dict)

    def set_optimal_params(self, optimal_params):
        """Inject optimal parameters recovered from offline system identification.

        Parameters
        ----------
        optimal_params : Dict[str, Any]
            Dictionary stored in `optimal_params.pkl`; expected to contain a
            `global_spring_Y` key representing the globally identified Young's modulus.

        Side Effects
        ------------
        * Inserts the renamed Young's modulus under `init_spring_Y` so the simulator
          uses it for initialisation and optimisation warm starts.
        """

        optimal_params["init_spring_Y"] = optimal_params.pop("global_spring_Y")
        self.update_from_dict(optimal_params)


cfg = Config()
