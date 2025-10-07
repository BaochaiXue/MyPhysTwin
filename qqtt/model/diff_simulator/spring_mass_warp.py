# Stage 2—Differentiable Physics Core
# Role: Implement Warp kernels for spring-force evaluation, integration, collision
#       handling, and loss accumulation used throughout training and interactive runs.
# Inputs: Vertex positions, spring topology, physical parameters from
#         `InvPhyTrainerWarp`, ground-truth trajectories (`cfg`-provided tensors).
# Outputs: Updated simulator state, accumulated losses, gradients via Warp tapes.
# Key in-house deps: `qqtt.utils.cfg` for hyperparameters, project logger for debug,
#                    `SpringMassSystemWarp` consumers in trainer module.
# Side effects: Allocates Warp arrays on CUDA device, mutates shared simulator state,
#               exposes autograd tapes for PyTorch interop; no filesystem I/O.
# Assumptions: CUDA available, Warp initialised, input tensors sized consistently with
#              connectivity; collision hash grids sized to avoid overflow.

import torch
from qqtt.utils import logger, cfg
import warp as wp

wp.init()
wp.set_device("cuda:0")
if not cfg.use_graph:
    wp.config.mode = "debug"
    wp.config.verbose = True
    wp.config.verify_autograd_array_access = True


class State:
    """Container for per-substep simulator state stored as Warp arrays.

    The simulator snapshots one `State` per substep so that reverse-mode adjoints can
    traverse the integration sequence. Each attribute mirrors a physical quantity:
    positions, velocities before/after collisions, accumulated forces, and controller
    poses.
    """

    def __init__(self, wp_init_vertices, num_control_points):
        """Allocate Warp buffers for vertex and controller state.

        Parameters
        ----------
        wp_init_vertices : wp.array
            Prototype Warp array containing the vertex layout (and dtype) used to
            initialise the positions; `wp.zeros_like` clones its metadata.
        num_control_points : int
            Number of controller endpoints; controller arrays are sized accordingly and
            created without gradients because control inputs are treated as exogenous.
        """
        self.wp_x = wp.zeros_like(wp_init_vertices, requires_grad=True)
        self.wp_v_before_collision = wp.zeros_like(wp_init_vertices, requires_grad=True)
        self.wp_v_before_ground = wp.zeros_like(wp_init_vertices, requires_grad=True)
        self.wp_v = wp.zeros_like(self.wp_x, requires_grad=True)
        self.wp_vertice_forces = wp.zeros_like(self.wp_x, requires_grad=True)
        # No need to compute the gradient for the control points
        self.wp_control_x = wp.zeros(
            (num_control_points), dtype=wp.vec3, requires_grad=False
        )
        self.wp_control_v = wp.zeros_like(self.wp_control_x, requires_grad=False)

    def clear_forces(self):
        """Zero-out accumulated vertex forces before the next integration step."""
        self.wp_vertice_forces.zero_()

    # This takes more time but not necessary, will be overwritten directly
    # def clear_control(self):
    #     self.wp_control_x.zero_()
    #     self.wp_control_v.zero_()

    # def clear_states(self):
    #     self.wp_x.zero_()
    #     self.wp_v_before_ground.zero_()
    #     self.wp_v.zero_()

    @property
    def requires_grad(self):
        """Indicate whether automatic differentiation is active for the state tensors."""
        return self.wp_x.requires_grad


@wp.kernel(enable_backward=False)
def copy_vec3(data: wp.array(dtype=wp.vec3), origin: wp.array(dtype=wp.vec3)):
    """Element-wise copy of vec3 arrays, used when resetting Warp buffers.

    Parameters
    ----------
    data : wp.array(dtype=wp.vec3)
        Source vector array (`N x 3`) whose contents should be replicated.
    origin : wp.array(dtype=wp.vec3)
        Destination array mutated in-place to match `data`.
    """
    tid = wp.tid()
    origin[tid] = data[tid]


@wp.kernel(enable_backward=False)
def copy_int(data: wp.array(dtype=wp.int32), origin: wp.array(dtype=wp.int32)):
    """Copy integer arrays between Warp buffers without tracking gradients.

    Parameters
    ----------
    data : wp.array(dtype=wp.int32)
        Source integer array.
    origin : wp.array(dtype=wp.int32)
        Destination array receiving the copied values.
    """
    tid = wp.tid()
    origin[tid] = data[tid]


@wp.kernel(enable_backward=False)
def copy_float(data: wp.array(dtype=wp.float32), origin: wp.array(dtype=wp.float32)):
    """Copy float arrays—used for masses, rest lengths, etc.

    Parameters
    ----------
    data : wp.array(dtype=wp.float32)
        Source float array.
    origin : wp.array(dtype=wp.float32)
        Destination array updated with `data` values.
    """
    tid = wp.tid()
    origin[tid] = data[tid]


@wp.kernel(enable_backward=False)
def set_control_points(
    num_substeps: int,
    original_control_point: wp.array(dtype=wp.vec3),
    target_control_point: wp.array(dtype=wp.vec3),
    step: int,
    control_x: wp.array(dtype=wp.vec3),
):
    """Linearly interpolate controller positions across substeps for smooth motion.

    Parameters
    ----------
    num_substeps : int
        Number of physics substeps per frame; denominator for interpolation factor.
    original_control_point : wp.array(dtype=wp.vec3)
        Controller positions at the previous frame.
    target_control_point : wp.array(dtype=wp.vec3)
        Controller goal positions at the current frame.
    step : int
        Index of the current substep (0-based) within the frame.
    control_x : wp.array(dtype=wp.vec3)
        Output buffer storing interpolated controller positions for this substep.
    """
    # Set the control points in each substep
    tid = wp.tid()

    t = float(step + 1) / float(num_substeps)
    control_x[tid] = (
        original_control_point[tid]
        + (target_control_point[tid] - original_control_point[tid]) * t
    )


@wp.kernel
def eval_springs(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    control_x: wp.array(dtype=wp.vec3),
    control_v: wp.array(dtype=wp.vec3),
    num_object_points: int,
    springs: wp.array(dtype=wp.vec2i),
    rest_lengths: wp.array(dtype=float),
    spring_Y: wp.array(dtype=float),
    dashpot_damping: float,
    spring_Y_min: float,
    spring_Y_max: float,
    f: wp.array(dtype=wp.vec3),
):
    """Evaluate Hookean spring and dashpot forces between connected vertices.

    Parameters
    ----------
    x, v : wp.array(dtype=wp.vec3)
        Object vertex positions/velocities (`N x 3`).
    control_x, control_v : wp.array(dtype=wp.vec3)
        Controller positions/velocities appended to the vertex set.
    num_object_points : int
        Number of entries in `x` corresponding to object vertices (excludes controllers).
    springs : wp.array(dtype=wp.vec2i)
        `(E, 2)` spring index pairs referencing either object or controller vertices.
    rest_lengths : wp.array(dtype=float)
        Rest lengths for each spring edge.
    spring_Y : wp.array(dtype=float)
        Log-scale Young's modulus values per spring (optimised parameter).
    dashpot_damping : float
        Damping coefficient applied to relative velocities along spring directions.
    spring_Y_min, spring_Y_max : float
        Bounds for clamping exponentiated stiffness values.
    f : wp.array(dtype=wp.vec3)
        Output buffer accumulating net forces per object vertex.
    """
    tid = wp.tid()

    if wp.exp(spring_Y[tid]) > spring_Y_min:

        idx1 = springs[tid][0]
        idx2 = springs[tid][1]

        if idx1 >= num_object_points:
            x1 = control_x[idx1 - num_object_points]
            v1 = control_v[idx1 - num_object_points]
        else:
            x1 = x[idx1]
            v1 = v[idx1]
        if idx2 >= num_object_points:
            x2 = control_x[idx2 - num_object_points]
            v2 = control_v[idx2 - num_object_points]
        else:
            x2 = x[idx2]
            v2 = v[idx2]

        rest = rest_lengths[tid]

        dis = x2 - x1
        dis_len = wp.length(dis)

        d = dis / wp.max(dis_len, 1e-6)

        spring_force = (
            wp.clamp(wp.exp(spring_Y[tid]), low=spring_Y_min, high=spring_Y_max)
            * (dis_len / rest - 1.0)
            * d
        )

        v_rel = wp.dot(v2 - v1, d)
        dashpot_forces = dashpot_damping * v_rel * d

        overall_force = spring_force + dashpot_forces

        if idx1 < num_object_points:
            wp.atomic_add(f, idx1, overall_force)
        if idx2 < num_object_points:
            wp.atomic_sub(f, idx2, overall_force)


@wp.kernel
def update_vel_from_force(
    v: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    dt: float,
    drag_damping: float,
    reverse_factor: float,
    v_new: wp.array(dtype=wp.vec3),
):
    """Integrate velocities using accumulated forces, gravity, and drag.

    Parameters
    ----------
    v : wp.array(dtype=wp.vec3)
        Current velocities for object vertices.
    f : wp.array(dtype=wp.vec3)
        Accumulated forces produced by springs/collisions.
    masses : wp.array(dtype=wp.float32)
        Per-vertex masses.
    dt : float
        Integration timestep.
    drag_damping : float
        Aerodynamic drag coefficient.
    reverse_factor : float
        Sign flip applied when simulations operate in mirrored coordinate systems.
    v_new : wp.array(dtype=wp.vec3)
        Output velocities after integration.
    """
    tid = wp.tid()

    v0 = v[tid]
    f0 = f[tid]
    m0 = masses[tid]

    drag_damping_factor = wp.exp(-dt * drag_damping)
    all_force = f0 + m0 * wp.vec3(0.0, 0.0, -9.8) * reverse_factor
    a = all_force / m0
    v1 = v0 + a * dt
    v2 = v1 * drag_damping_factor

    v_new[tid] = v2


@wp.func
def loop(
    i: int,
    collision_indices: wp.array2d(dtype=wp.int32),
    collision_number: wp.array(dtype=wp.int32),
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    masks: wp.array(dtype=wp.int32),
    collision_dist: float,
    clamp_collide_object_elas: float,
    clamp_collide_object_fric: float,
):
    """Aggregate collision impulses for point `i` against nearby contact candidates.

    Parameters
    ----------
    i : int
        Index of the vertex being processed.
    collision_indices : wp.array2d(dtype=wp.int32)
        Precomputed neighbour indices for potential collisions (shape `N x K`).
    collision_number : wp.array(dtype=wp.int32)
        Count of valid neighbours per vertex.
    x, v : wp.array(dtype=wp.vec3)
        Positions and velocities of object vertices.
    masses : wp.array(dtype=wp.float32)
        Per-vertex masses.
    masks : wp.array(dtype=wp.int32)
        Object identifiers used to avoid self-collisions.
    collision_dist : float
        Maximum separation for a collision to be considered.
    clamp_collide_object_elas, clamp_collide_object_fric : float
        Clamped restitution and friction coefficients for object-object collisions.

    Returns
    -------
    Tuple[float, wp.vec3]
        Count of valid collisions and the summed impulse vector.
    """
    x1 = x[i]
    v1 = v[i]
    m1 = masses[i]
    mask1 = masks[i]

    valid_count = float(0.0)
    J_sum = wp.vec3(0.0, 0.0, 0.0)
    for k in range(collision_number[i]):
        index = collision_indices[i][k]
        x2 = x[index]
        v2 = v[index]
        m2 = masses[index]
        mask2 = masks[index]

        dis = x2 - x1
        dis_len = wp.length(dis)
        relative_v = v2 - v1
        # If the distance is less than the collision distance and the two points are moving towards each other
        if (
            mask1 != mask2
            and dis_len < collision_dist
            and wp.dot(dis, relative_v) < -1e-4
        ):
            valid_count += 1.0

            collision_normal = dis / wp.max(dis_len, 1e-6)
            v_rel_n = wp.dot(relative_v, collision_normal) * collision_normal
            impulse_n = (-(1.0 + clamp_collide_object_elas) * v_rel_n) / (
                1.0 / m1 + 1.0 / m2
            )
            v_rel_n_length = wp.length(v_rel_n)

            v_rel_t = relative_v - v_rel_n
            v_rel_t_length = wp.max(wp.length(v_rel_t), 1e-6)
            a = wp.max(
                0.0,
                1.0
                - clamp_collide_object_fric
                * (1.0 + clamp_collide_object_elas)
                * v_rel_n_length
                / v_rel_t_length,
            )
            impulse_t = (a - 1.0) * v_rel_t / (1.0 / m1 + 1.0 / m2)

            J = impulse_n + impulse_t

            J_sum += J

    return valid_count, J_sum


@wp.kernel(enable_backward=False)
def update_potential_collision(
    x: wp.array(dtype=wp.vec3),
    masks: wp.array(dtype=wp.int32),
    collision_dist: float,
    grid: wp.uint64,
    collision_indices: wp.array2d(dtype=wp.int32),
    collision_number: wp.array(dtype=wp.int32),
):
    """Populate per-vertex candidate lists for object-object collision checks.

    Parameters
    ----------
    x : wp.array(dtype=wp.vec3)
        Object vertex positions (`N x 3`).
    masks : wp.array(dtype=wp.int32)
        Object identifiers used to skip intra-object collisions.
    collision_dist : float
        Maximum search radius for potential collisions.
    grid : wp.uint64
        Hash-grid handle storing spatial partitioning of vertices.
    collision_indices : wp.array2d(dtype=wp.int32)
        Output array capturing neighbour indices for each vertex.
    collision_number : wp.array(dtype=wp.int32)
        Output counts of neighbours per vertex.
    """
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    x1 = x[i]
    mask1 = masks[i]

    neighbors = wp.hash_grid_query(grid, x1, collision_dist * 5.0)
    for index in neighbors:
        if index != i:
            x2 = x[index]
            mask2 = masks[index]

            dis = x2 - x1
            dis_len = wp.length(dis)
            # If the distance is less than the collision distance and the two points are moving towards each other
            if mask1 != mask2 and dis_len < collision_dist:
                collision_indices[i][collision_number[i]] = index
                collision_number[i] += 1


@wp.kernel
def object_collision(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    masks: wp.array(dtype=wp.int32),
    collide_object_elas: wp.array(dtype=float),
    collide_object_fric: wp.array(dtype=float),
    collision_dist: float,
    collision_indices: wp.array2d(dtype=wp.int32),
    collision_number: wp.array(dtype=wp.int32),
    v_new: wp.array(dtype=wp.vec3),
):
    """Resolve pairwise object collisions via impulse-based response.

    Parameters
    ----------
    x, v : wp.array(dtype=wp.vec3)
        Positions and velocities of object vertices.
    masses : wp.array(dtype=wp.float32)
        Per-vertex masses governing impulse magnitude.
    masks : wp.array(dtype=wp.int32)
        Object identifiers to prevent self-collisions.
    collide_object_elas, collide_object_fric : wp.array(dtype=float)
        Scalar buffers storing restitution and friction coefficients.
    collision_dist : float
        Maximum distance for collisions to be evaluated (used in the candidate set).
    collision_indices, collision_number : see `update_potential_collision`.
    v_new : wp.array(dtype=wp.vec3)
        Output velocities after applying averaged impulses.
    """
    tid = wp.tid()

    v1 = v[tid]
    m1 = masses[tid]

    clamp_collide_object_elas = wp.clamp(collide_object_elas[0], low=0.0, high=1.0)
    clamp_collide_object_fric = wp.clamp(collide_object_fric[0], low=0.0, high=2.0)

    valid_count, J_sum = loop(
        tid,
        collision_indices,
        collision_number,
        x,
        v,
        masses,
        masks,
        collision_dist,
        clamp_collide_object_elas,
        clamp_collide_object_fric,
    )

    if valid_count > 0:
        J_average = J_sum / valid_count
        v_new[tid] = v1 - J_average / m1
    else:
        v_new[tid] = v1


@wp.kernel
def integrate_ground_collision(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    collide_elas: wp.array(dtype=float),
    collide_fric: wp.array(dtype=float),
    dt: float,
    reverse_factor: float,
    x_new: wp.array(dtype=wp.vec3),
    v_new: wp.array(dtype=wp.vec3),
):
    """Handle collisions against an infinite ground plane with restitution/friction.

    Parameters
    ----------
    x, v : wp.array(dtype=wp.vec3)
        Vertex positions and velocities.
    collide_elas, collide_fric : wp.array(dtype=float)
        Scalars storing ground restitution/friction coefficients.
    dt : float
        Simulation timestep.
    reverse_factor : float
        Sign flip applied when sim axes are reversed.
    x_new, v_new : wp.array(dtype=wp.vec3)
        Output buffers storing updated positions and velocities.
    """
    tid = wp.tid()

    x0 = x[tid]
    v0 = v[tid]

    normal = wp.vec3(0.0, 0.0, 1.0) * reverse_factor

    x_z = x0[2]
    v_z = v0[2]
    next_x_z = (x_z + v_z * dt) * reverse_factor

    if next_x_z < 0.0 and v_z * reverse_factor < -1e-4:
        # Ground Collision
        v_normal = wp.dot(v0, normal) * normal
        v_tao = v0 - v_normal
        v_normal_length = wp.length(v_normal)
        v_tao_length = wp.max(wp.length(v_tao), 1e-6)
        clamp_collide_elas = wp.clamp(collide_elas[0], low=0.0, high=1.0)
        clamp_collide_fric = wp.clamp(collide_fric[0], low=0.0, high=2.0)

        v_normal_new = -clamp_collide_elas * v_normal
        a = wp.max(
            0.0,
            1.0
            - clamp_collide_fric
            * (1.0 + clamp_collide_elas)
            * v_normal_length
            / v_tao_length,
        )
        v_tao_new = a * v_tao

        v1 = v_normal_new + v_tao_new
        toi = -x_z / v_z
    else:
        v1 = v0
        toi = 0.0

    x_new[tid] = x0 + v0 * toi + v1 * (dt - toi)
    v_new[tid] = v1


@wp.kernel(enable_backward=False)
def compute_distances(
    pred: wp.array(dtype=wp.vec3),
    gt: wp.array(dtype=wp.vec3),
    gt_mask: wp.array(dtype=wp.int32),
    distances: wp.array2d(dtype=float),
):
    """Compute masked pairwise distances between predicted and ground-truth points.

    Parameters
    ----------
    pred : wp.array(dtype=wp.vec3)
        Predicted vertex positions (`N x 3`).
    gt : wp.array(dtype=wp.vec3)
        Ground-truth positions (`M x 3`).
    gt_mask : wp.array(dtype=wp.int32)
        Visibility mask; zero entries skip the corresponding GT vertex.
    distances : wp.array2d(dtype=float)
        Output distance matrix (`M x N`).
    """
    i, j = wp.tid()
    if gt_mask[i] == 1:
        dist = wp.length(gt[i] - pred[j])
        distances[i, j] = dist
    else:
        distances[i, j] = 1e6


@wp.kernel(enable_backward=False)
def compute_neigh_indices(
    distances: wp.array2d(dtype=float),
    neigh_indices: wp.array(dtype=wp.int32),
):
    """Identify the closest predicted neighbour index for each GT vertex.

    Parameters
    ----------
    distances : wp.array2d(dtype=float)
        Pairwise distance matrix generated by `compute_distances`.
    neigh_indices : wp.array(dtype=wp.int32)
        Output indices of the nearest predicted vertex per GT point.
    """
    i = wp.tid()
    min_dist = float(1e6)
    min_index = int(-1)
    for j in range(distances.shape[1]):
        if distances[i, j] < min_dist:
            min_dist = distances[i, j]
            min_index = j
    neigh_indices[i] = min_index


@wp.kernel
def compute_chamfer_loss(
    pred: wp.array(dtype=wp.vec3),
    gt: wp.array(dtype=wp.vec3),
    gt_mask: wp.array(dtype=wp.int32),
    num_valid: int,
    neigh_indices: wp.array(dtype=wp.int32),
    loss_weight: float,
    chamfer_loss: wp.array(dtype=float),
):
    """Accumulate weighted squared distances for the Chamfer loss term.

    Parameters
    ----------
    pred, gt : wp.array(dtype=wp.vec3)
        Predicted and ground-truth positions.
    gt_mask : wp.array(dtype=wp.int32)
        Visibility mask for ground-truth points.
    num_valid : int
        Count of visible GT points (normalises the loss).
    neigh_indices : wp.array(dtype=wp.int32)
        Mapped nearest-neighbour indices per GT point.
    loss_weight : float
        Multiplier applied to the Chamfer term.
    chamfer_loss : wp.array(dtype=float)
        Scalar accumulator storing the resulting loss.
    """
    i = wp.tid()
    if gt_mask[i] == 1:
        min_pred = pred[neigh_indices[i]]
        min_dist = wp.length(min_pred - gt[i])
        final_min_dist = loss_weight * min_dist * min_dist / float(num_valid)
        wp.atomic_add(chamfer_loss, 0, final_min_dist)


@wp.kernel
def compute_track_loss(
    pred: wp.array(dtype=wp.vec3),
    gt: wp.array(dtype=wp.vec3),
    gt_mask: wp.array(dtype=wp.int32),
    num_valid: int,
    loss_weight: float,
    track_loss: wp.array(dtype=float),
):
    """Compute smooth-L1 positional error for tracked, visible points.

    Parameters
    ----------
    pred, gt : wp.array(dtype=wp.vec3)
        Predicted and ground-truth positions.
    gt_mask : wp.array(dtype=wp.int32)
        Visibility mask.
    num_valid : int
        Number of visible points (for averaging).
    loss_weight : float
        Multiplier applied to the tracking loss term.
    track_loss : wp.array(dtype=float)
        Scalar accumulator storing the loss value.
    """
    i = wp.tid()
    if gt_mask[i] == 1:
        # Calculate the smooth l1 loss modifed from fvcore.nn.smooth_l1_loss
        pred_x = pred[i][0]
        pred_y = pred[i][1]
        pred_z = pred[i][2]
        gt_x = gt[i][0]
        gt_y = gt[i][1]
        gt_z = gt[i][2]

        dist_x = wp.abs(pred_x - gt_x)
        dist_y = wp.abs(pred_y - gt_y)
        dist_z = wp.abs(pred_z - gt_z)

        if dist_x < 1.0:
            temp_track_loss_x = 0.5 * (dist_x**2.0)
        else:
            temp_track_loss_x = dist_x - 0.5

        if dist_y < 1.0:
            temp_track_loss_y = 0.5 * (dist_y**2.0)
        else:
            temp_track_loss_y = dist_y - 0.5

        if dist_z < 1.0:
            temp_track_loss_z = 0.5 * (dist_z**2.0)
        else:
            temp_track_loss_z = dist_z - 0.5

        temp_track_loss = temp_track_loss_x + temp_track_loss_y + temp_track_loss_z

        average_factor = float(num_valid) * 3.0

        final_track_loss = loss_weight * temp_track_loss / average_factor

        wp.atomic_add(track_loss, 0, final_track_loss)


@wp.kernel(enable_backward=False)
def set_int(input: int, output: wp.array(dtype=wp.int32)):
    """Write scalar integer values into Warp arrays (used for counters).

    Parameters
    ----------
    input : int
        Value to write into the array.
    output : wp.array(dtype=wp.int32)
        Destination array whose first element is set to `input`.
    """
    output[0] = input


@wp.kernel(enable_backward=False)
def update_acc(
    v1: wp.array(dtype=wp.vec3),
    v2: wp.array(dtype=wp.vec3),
    prev_acc: wp.array(dtype=wp.vec3),
):
    """Store previous-step accelerations for temporal smoothness regularisation.

    Parameters
    ----------
    v1, v2 : wp.array(dtype=wp.vec3)
        Previous and current velocities.
    prev_acc : wp.array(dtype=wp.vec3)
        Output buffer storing last-step acceleration (used in loss computation).
    """
    tid = wp.tid()
    prev_acc[tid] = v2[tid] - v1[tid]


@wp.kernel
def compute_acc_loss(
    v1: wp.array(dtype=wp.vec3),
    v2: wp.array(dtype=wp.vec3),
    prev_acc: wp.array(dtype=wp.vec3),
    num_object_points: int,
    acc_count: wp.array(dtype=wp.int32),
    acc_weight: float,
    acc_loss: wp.array(dtype=wp.float32),
):
    """Compute smooth-L1 penalty on acceleration changes to discourage jitter.

    Parameters
    ----------
    v1, v2 : wp.array(dtype=wp.vec3)
        Velocities at consecutive frames.
    prev_acc : wp.array(dtype=wp.vec3)
        Cached accelerations from the previous frame.
    num_object_points : int
        Number of vertices considered when averaging the loss.
    acc_count : wp.array(dtype=wp.int32)
        Flag to toggle the acceleration loss (0/1).
    acc_weight : float
        Multiplier for the acceleration loss term.
    acc_loss : wp.array(dtype=wp.float32)
        Scalar accumulator storing the loss value.
    """
    if acc_count[0] == 1:
        # Calculate the smooth l1 loss modifed from fvcore.nn.smooth_l1_loss
        tid = wp.tid()
        cur_acc = v2[tid] - v1[tid]
        cur_x = cur_acc[0]
        cur_y = cur_acc[1]
        cur_z = cur_acc[2]

        prev_x = prev_acc[tid][0]
        prev_y = prev_acc[tid][1]
        prev_z = prev_acc[tid][2]

        dist_x = wp.abs(cur_x - prev_x)
        dist_y = wp.abs(cur_y - prev_y)
        dist_z = wp.abs(cur_z - prev_z)

        if dist_x < 1.0:
            temp_acc_loss_x = 0.5 * (dist_x**2.0)
        else:
            temp_acc_loss_x = dist_x - 0.5

        if dist_y < 1.0:
            temp_acc_loss_y = 0.5 * (dist_y**2.0)
        else:
            temp_acc_loss_y = dist_y - 0.5

        if dist_z < 1.0:
            temp_acc_loss_z = 0.5 * (dist_z**2.0)
        else:
            temp_acc_loss_z = dist_z - 0.5

        temp_acc_loss = temp_acc_loss_x + temp_acc_loss_y + temp_acc_loss_z

        average_factor = float(num_object_points) * 3.0

        final_acc_loss = acc_weight * temp_acc_loss / average_factor

        wp.atomic_add(acc_loss, 0, final_acc_loss)


@wp.kernel
def compute_final_loss(
    chamfer_loss: wp.array(dtype=wp.float32),
    track_loss: wp.array(dtype=wp.float32),
    acc_loss: wp.array(dtype=wp.float32),
    loss: wp.array(dtype=wp.float32),
):
    """Combine individual loss terms into a scalar objective.

    Parameters
    ----------
    chamfer_loss, track_loss, acc_loss : wp.array(dtype=wp.float32)
        Individual loss accumulators.
    loss : wp.array(dtype=wp.float32)
        Output scalar storing the sum of all terms.
    """
    loss[0] = chamfer_loss[0] + track_loss[0] + acc_loss[0]


@wp.kernel
def compute_simple_loss(
    pred: wp.array(dtype=wp.vec3),
    gt: wp.array(dtype=wp.vec3),
    num_object_points: int,
    loss: wp.array(dtype=wp.float32),
):
    """Compute smooth-L1 reconstruction loss for synthetic datasets.

    Parameters
    ----------
    pred, gt : wp.array(dtype=wp.vec3)
        Predicted and ground-truth positions.
    num_object_points : int
        Number of vertices (used for averaging).
    loss : wp.array(dtype=wp.float32)
        Scalar accumulator storing the resulting loss.
    """
    # Calculate the smooth l1 loss modifed from fvcore.nn.smooth_l1_loss
    tid = wp.tid()
    pred_x = pred[tid][0]
    pred_y = pred[tid][1]
    pred_z = pred[tid][2]

    gt_x = gt[tid][0]
    gt_y = gt[tid][1]
    gt_z = gt[tid][2]

    dist_x = wp.abs(pred_x - gt_x)
    dist_y = wp.abs(pred_y - gt_y)
    dist_z = wp.abs(pred_z - gt_z)

    if dist_x < 1.0:
        temp_simple_loss_x = 0.5 * (dist_x**2.0)
    else:
        temp_simple_loss_x = dist_x - 0.5

    if dist_y < 1.0:
        temp_simple_loss_y = 0.5 * (dist_y**2.0)
    else:
        temp_simple_loss_y = dist_y - 0.5

    if dist_z < 1.0:
        temp_simple_loss_z = 0.5 * (dist_z**2.0)
    else:
        temp_simple_loss_z = dist_z - 0.5

    temp_simple_loss = temp_simple_loss_x + temp_simple_loss_y + temp_simple_loss_z

    average_factor = float(num_object_points) * 3.0

    final_simple_loss = temp_simple_loss / average_factor

    wp.atomic_add(loss, 0, final_simple_loss)


class SpringMassSystemWarp:
    """Differentiable spring-mass simulator with collision handling and losses."""
    def __init__(
        self,
        init_vertices,
        init_springs,
        init_rest_lengths,
        init_masses,
        dt,
        num_substeps,
        spring_Y,
        collide_elas,
        collide_fric,
        dashpot_damping,
        drag_damping,
        collide_object_elas=0.7,
        collide_object_fric=0.3,
        init_masks=None,
        collision_dist=0.02,
        init_velocities=None,
        num_object_points=None,
        num_surface_points=None,
        num_original_points=None,
        controller_points=None,
        reverse_z=False,
        spring_Y_min=1e3,
        spring_Y_max=1e5,
        gt_object_points=None,
        gt_object_visibilities=None,
        gt_object_motions_valid=None,
        self_collision=False,
        disable_backward=False,
    ):
        """Instantiate the Warp simulator with mesh topology and physical parameters.

        Parameters
        ----------
        init_vertices : torch.Tensor
            `(N, 3)` initial vertex positions combining object, controller, and interior
            points.
        init_springs : torch.Tensor
            `(E, 2)` integer pairs defining spring connectivity between vertices.
        init_rest_lengths : torch.Tensor
            `(E,)` rest distances for each spring edge.
        init_masses : torch.Tensor
            `(N,)` per-vertex masses used during integration.
        dt : float
            Simulation timestep in seconds.
        num_substeps : int
            Number of substeps to execute per frame for stability.
        spring_Y : float
            Initial Young's modulus; optimised during training.
        collide_elas, collide_fric : float
            Ground collision restitution and friction coefficients.
        dashpot_damping, drag_damping : float
            Damping coefficients for springs and air drag.
        collide_object_elas, collide_object_fric : float
            Object-object collision parameters.
        init_masks : torch.Tensor, optional
            Per-vertex object identifiers to limit collision pairs.
        collision_dist : float
            Distance threshold for triggering object-object collisions.
        init_velocities : torch.Tensor, optional
            `(N, 3)` initial velocity field.
        num_object_points, num_surface_points, num_original_points : int
            Bookkeeping counts to distinguish object vs. controller vertices.
        controller_points : torch.Tensor, optional
            `(F, C, 3)` controller trajectories for real datasets.
        reverse_z : bool
            Flip gravity axis to align with dataset coordinate conventions.
        spring_Y_min, spring_Y_max : float
            Bounds applied to exponential Young's modulus parameterisation.
        gt_object_points, gt_object_visibilities, gt_object_motions_valid : torch.Tensor
            Ground-truth trajectories and masks for loss computation.
        self_collision : bool
            If True, enable collision among all vertices (treated as unique masks).
        disable_backward : bool
            Skip gradient capture when building Warp graphs for inference-only mode.
        """
        logger.info(f"[SIMULATION]: Initialize the Spring-Mass System")
        self.device = cfg.device

        # Record the parameters
        self.wp_init_vertices = wp.from_torch(
            init_vertices[:num_object_points].contiguous(),
            dtype=wp.vec3,
            requires_grad=False,
        )
        if init_velocities is None:
            self.wp_init_velocities = wp.zeros_like(
                self.wp_init_vertices, requires_grad=False
            )
        else:
            self.wp_init_velocities = wp.from_torch(
                init_velocities[:num_object_points].contiguous(),
                dtype=wp.vec3,
                requires_grad=False,
            )

        self.n_vertices = init_vertices.shape[0]
        self.n_springs = init_springs.shape[0]

        self.dt = dt
        self.num_substeps = num_substeps
        self.dashpot_damping = dashpot_damping
        self.drag_damping = drag_damping
        self.reverse_factor = 1.0 if not reverse_z else -1.0
        self.spring_Y_min = spring_Y_min
        self.spring_Y_max = spring_Y_max

        if controller_points is None:
            assert num_object_points == self.n_vertices
        else:
            assert (controller_points.shape[1] + num_object_points) == self.n_vertices
        self.num_object_points = num_object_points
        self.num_control_points = (
            controller_points.shape[1] if not controller_points is None else 0
        )
        self.controller_points = controller_points

        # Deal with the any collision detection
        self.object_collision_flag = 0
        if init_masks is not None:
            if torch.unique(init_masks).shape[0] > 1:
                self.object_collision_flag = 1

        if self_collision:
            assert init_masks is None
            self.object_collision_flag = 1
            # Make all points as the collision points
            init_masks = torch.arange(
                self.n_vertices, dtype=torch.int32, device=self.device
            )

        if self.object_collision_flag:
            self.wp_masks = wp.from_torch(
                init_masks[:num_object_points].int(),
                dtype=wp.int32,
                requires_grad=False,
            )

            self.collision_grid = wp.HashGrid(128, 128, 128)
            self.collision_dist = collision_dist

            self.wp_collision_indices = wp.zeros(
                (self.wp_init_vertices.shape[0], 500),
                dtype=wp.int32,
                requires_grad=False,
            )
            self.wp_collision_number = wp.zeros(
                (self.wp_init_vertices.shape[0]), dtype=wp.int32, requires_grad=False
            )

        # Initialize the GT for calculating losses
        self.gt_object_points = gt_object_points
        if cfg.data_type == "real":
            self.gt_object_visibilities = gt_object_visibilities.int()
            self.gt_object_motions_valid = gt_object_motions_valid.int()

        self.num_surface_points = num_surface_points
        self.num_original_points = num_original_points
        if num_original_points is None:
            self.num_original_points = self.num_object_points

        # # Do some initialization to initialize the warp cuda graph
        self.wp_springs = wp.from_torch(
            init_springs, dtype=wp.vec2i, requires_grad=False
        )
        self.wp_rest_lengths = wp.from_torch(
            init_rest_lengths, dtype=wp.float32, requires_grad=False
        )
        self.wp_masses = wp.from_torch(
            init_masses[:num_object_points], dtype=wp.float32, requires_grad=False
        )
        if cfg.data_type == "real":
            self.prev_acc = wp.zeros_like(self.wp_init_vertices, requires_grad=False)
            self.acc_count = wp.zeros(1, dtype=wp.int32, requires_grad=False)

        self.wp_current_object_points = wp.from_torch(
            self.gt_object_points[1].clone(), dtype=wp.vec3, requires_grad=False
        )
        if cfg.data_type == "real":
            self.wp_current_object_visibilities = wp.from_torch(
                self.gt_object_visibilities[1].clone(),
                dtype=wp.int32,
                requires_grad=False,
            )
            self.wp_current_object_motions_valid = wp.from_torch(
                self.gt_object_motions_valid[0].clone(),
                dtype=wp.int32,
                requires_grad=False,
            )
            self.num_valid_visibilities = int(self.gt_object_visibilities[1].sum())
            self.num_valid_motions = int(self.gt_object_motions_valid[0].sum())

            self.wp_original_control_point = wp.from_torch(
                self.controller_points[0].clone(), dtype=wp.vec3, requires_grad=False
            )
            self.wp_target_control_point = wp.from_torch(
                self.controller_points[1].clone(), dtype=wp.vec3, requires_grad=False
            )

            self.chamfer_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            self.track_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            self.acc_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

        # Initialize the warp parameters
        self.wp_states = []
        for i in range(self.num_substeps + 1):
            state = State(self.wp_init_velocities, self.num_control_points)
            self.wp_states.append(state)
        if cfg.data_type == "real":
            self.distance_matrix = wp.zeros(
                (self.num_original_points, self.num_surface_points), requires_grad=False
            )
            self.neigh_indices = wp.zeros(
                (self.num_original_points), dtype=wp.int32, requires_grad=False
            )

        # Parameter to be optimized
        self.wp_spring_Y = wp.from_torch(
            torch.log(torch.tensor(spring_Y, dtype=torch.float32, device=self.device))
            * torch.ones(self.n_springs, dtype=torch.float32, device=self.device),
            requires_grad=True,
        )
        self.wp_collide_elas = wp.from_torch(
            torch.tensor([collide_elas], dtype=torch.float32, device=self.device),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_fric = wp.from_torch(
            torch.tensor([collide_fric], dtype=torch.float32, device=self.device),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_object_elas = wp.from_torch(
            torch.tensor(
                [collide_object_elas], dtype=torch.float32, device=self.device
            ),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_object_fric = wp.from_torch(
            torch.tensor(
                [collide_object_fric], dtype=torch.float32, device=self.device
            ),
            requires_grad=cfg.collision_learn,
        )

        # Create the CUDA graph to acclerate
        if cfg.use_graph:
            if cfg.data_type == "real":
                if not disable_backward:
                    with wp.ScopedCapture() as capture:
                        self.tape = wp.Tape()
                        with self.tape:
                            self.step()
                            self.calculate_loss()
                        self.tape.backward(self.loss)
                else:
                    with wp.ScopedCapture() as capture:
                        self.step()
                        self.calculate_loss()
                self.graph = capture.graph
            elif cfg.data_type == "synthetic":
                if not disable_backward:
                    # For synthetic data, we compute simple loss
                    with wp.ScopedCapture() as capture:
                        self.tape = wp.Tape()
                        with self.tape:
                            self.step()
                            self.calculate_simple_loss()
                        self.tape.backward(self.loss)
                else:
                    with wp.ScopedCapture() as capture:
                        self.step()
                        self.calculate_simple_loss()
                self.graph = capture.graph
            else:
                raise NotImplementedError

            with wp.ScopedCapture() as forward_capture:
                self.step()
            self.forward_graph = forward_capture.graph
        else:
            self.tape = wp.Tape()

    def set_controller_target(self, frame_idx, pure_inference=False):
        """Update controller Warp buffers using pre-recorded trajectories.

        Parameters
        ----------
        frame_idx : int
            Index of the dataset frame to sample controller/object targets from.
        pure_inference : bool, optional
            When True, skip loading ground-truth object trajectories/masks (used in
            inference-only loops).
        """
        if self.controller_points is not None:
            # Set the controller points
            wp.launch(
                copy_vec3,
                dim=self.num_control_points,
                inputs=[self.controller_points[frame_idx - 1]],
                outputs=[self.wp_original_control_point],
            )
            wp.launch(
                copy_vec3,
                dim=self.num_control_points,
                inputs=[self.controller_points[frame_idx]],
                outputs=[self.wp_target_control_point],
            )

        if not pure_inference:
            # Set the target points
            wp.launch(
                copy_vec3,
                dim=self.num_original_points,
                inputs=[self.gt_object_points[frame_idx]],
                outputs=[self.wp_current_object_points],
            )

            if cfg.data_type == "real":
                wp.launch(
                    copy_int,
                    dim=self.num_original_points,
                    inputs=[self.gt_object_visibilities[frame_idx]],
                    outputs=[self.wp_current_object_visibilities],
                )
                wp.launch(
                    copy_int,
                    dim=self.num_original_points,
                    inputs=[self.gt_object_motions_valid[frame_idx - 1]],
                    outputs=[self.wp_current_object_motions_valid],
                )

                self.num_valid_visibilities = int(
                    self.gt_object_visibilities[frame_idx].sum()
                )
                self.num_valid_motions = int(
                    self.gt_object_motions_valid[frame_idx - 1].sum()
                )

    def set_controller_interactive(
        self, last_controller_interactive, controller_interactive
    ):
        """Copy interactive controller poses from UI buffers into Warp arrays.

        Parameters
        ----------
        last_controller_interactive : torch.Tensor
            `(C, 3)` positions from the previous frame.
        controller_interactive : torch.Tensor
            `(C, 3)` current target positions produced by the UI.
        """
        # Set the controller points
        wp.launch(
            copy_vec3,
            dim=self.num_control_points,
            inputs=[last_controller_interactive],
            outputs=[self.wp_original_control_point],
        )
        wp.launch(
            copy_vec3,
            dim=self.num_control_points,
            inputs=[controller_interactive],
            outputs=[self.wp_target_control_point],
        )

    def set_init_state(self, wp_x, wp_v, pure_inference=False):
        """Populate the first state snapshot with positions/velocities for a rollout.

        Parameters
        ----------
        wp_x : wp.array(dtype=wp.vec3)
            Object vertex positions initialising the rollout.
        wp_v : wp.array(dtype=wp.vec3)
            Corresponding initial velocities.
        pure_inference : bool, optional
            Skip cloning states when running inside a captured CUDA graph for inference.
        """
        # Detach and clone and set requires_grad=True
        assert (
            self.num_object_points == wp_x.shape[0]
            and self.num_object_points == self.wp_states[0].wp_x.shape[0]
        )

        if not pure_inference:
            wp.launch(
                copy_vec3,
                dim=self.num_object_points,
                inputs=[wp.clone(wp_x, requires_grad=False)],
                outputs=[self.wp_states[0].wp_x],
            )
            wp.launch(
                copy_vec3,
                dim=self.num_object_points,
                inputs=[wp.clone(wp_v, requires_grad=False)],
                outputs=[self.wp_states[0].wp_v],
            )
        else:
            wp.launch(
                copy_vec3,
                dim=self.num_object_points,
                inputs=[wp_x],
                outputs=[self.wp_states[0].wp_x],
            )
            wp.launch(
                copy_vec3,
                dim=self.num_object_points,
                inputs=[wp_v],
                outputs=[self.wp_states[0].wp_v],
            )

    def set_acc_count(self, acc_count):
        """Toggle acceleration regularisation based on history availability.

        Parameters
        ----------
        acc_count : bool
            True when previous-frame accelerations are valid and the loss should be
            activated; False otherwise.
        """
        if acc_count:
            input = 1
        else:
            input = 0
        wp.launch(
            set_int,
            dim=1,
            inputs=[input],
            outputs=[self.acc_count],
        )

    def update_acc(self):
        """Launch the kernel that caches accelerations for the next timestep."""
        wp.launch(
            update_acc,
            dim=self.num_object_points,
            inputs=[
                wp.clone(self.wp_states[0].wp_v, requires_grad=False),
                wp.clone(self.wp_states[-1].wp_v, requires_grad=False),
            ],
            outputs=[self.prev_acc],
        )

    def update_collision_graph(self):
        """Rebuild the hash grid and neighbour lists for object-object collisions."""
        assert self.object_collision_flag
        self.collision_grid.build(self.wp_states[0].wp_x, self.collision_dist * 5.0)
        self.wp_collision_number.zero_()
        wp.launch(
            update_potential_collision,
            dim=self.num_object_points,
            inputs=[
                self.wp_states[0].wp_x,
                self.wp_masks,
                self.collision_dist,
                self.collision_grid.id,
            ],
            outputs=[self.wp_collision_indices, self.wp_collision_number],
        )

    def step(self):
        """Advance the simulator by one frame using semi-implicit integration."""
        for i in range(self.num_substeps):
            self.wp_states[i].clear_forces()
            if not self.controller_points is None:
                # Set the control point
                wp.launch(
                    set_control_points,
                    dim=self.num_control_points,
                    inputs=[
                        self.num_substeps,
                        self.wp_original_control_point,
                        self.wp_target_control_point,
                        i,
                    ],
                    outputs=[self.wp_states[i].wp_control_x],
                )

            # Calculate the spring forces
            wp.launch(
                kernel=eval_springs,
                dim=self.n_springs,
                inputs=[
                    self.wp_states[i].wp_x,
                    self.wp_states[i].wp_v,
                    self.wp_states[i].wp_control_x,
                    self.wp_states[i].wp_control_v,
                    self.num_object_points,
                    self.wp_springs,
                    self.wp_rest_lengths,
                    self.wp_spring_Y,
                    self.dashpot_damping,
                    self.spring_Y_min,
                    self.spring_Y_max,
                ],
                outputs=[self.wp_states[i].wp_vertice_forces],
            )

            if self.object_collision_flag:
                output_v = self.wp_states[i].wp_v_before_collision
            else:
                output_v = self.wp_states[i].wp_v_before_ground

            # Update the output_v using the vertive_forces
            wp.launch(
                kernel=update_vel_from_force,
                dim=self.num_object_points,
                inputs=[
                    self.wp_states[i].wp_v,
                    self.wp_states[i].wp_vertice_forces,
                    self.wp_masses,
                    self.dt,
                    self.drag_damping,
                    self.reverse_factor,
                ],
                outputs=[output_v],
            )

            if self.object_collision_flag:
                # Update the wp_v_before_ground based on the collision handling
                wp.launch(
                    kernel=object_collision,
                    dim=self.num_object_points,
                    inputs=[
                        self.wp_states[i].wp_x,
                        self.wp_states[i].wp_v_before_collision,
                        self.wp_masses,
                        self.wp_masks,
                        self.wp_collide_object_elas,
                        self.wp_collide_object_fric,
                        self.collision_dist,
                        self.wp_collision_indices,
                        self.wp_collision_number,
                    ],
                    outputs=[self.wp_states[i].wp_v_before_ground],
                )

            # Update the x and v
            wp.launch(
                kernel=integrate_ground_collision,
                dim=self.num_object_points,
                inputs=[
                    self.wp_states[i].wp_x,
                    self.wp_states[i].wp_v_before_ground,
                    self.wp_collide_elas,
                    self.wp_collide_fric,
                    self.dt,
                    self.reverse_factor,
                ],
                outputs=[self.wp_states[i + 1].wp_x, self.wp_states[i + 1].wp_v],
            )

    def calculate_loss(self):
        """Evaluate full loss terms (Chamfer, track, acceleration) for real data."""
        # Compute the chamfer loss
        # Precompute the distances matrix for the chamfer loss
        wp.launch(
            compute_distances,
            dim=(self.num_original_points, self.num_surface_points),
            inputs=[
                self.wp_states[-1].wp_x,
                self.wp_current_object_points,
                self.wp_current_object_visibilities,
            ],
            outputs=[self.distance_matrix],
        )

        wp.launch(
            compute_neigh_indices,
            dim=self.num_original_points,
            inputs=[self.distance_matrix],
            outputs=[self.neigh_indices],
        )

        wp.launch(
            compute_chamfer_loss,
            dim=self.num_original_points,
            inputs=[
                self.wp_states[-1].wp_x,
                self.wp_current_object_points,
                self.wp_current_object_visibilities,
                self.num_valid_visibilities,
                self.neigh_indices,
                cfg.chamfer_weight,
            ],
            outputs=[self.chamfer_loss],
        )

        # Compute the tracking loss
        wp.launch(
            compute_track_loss,
            dim=self.num_original_points,
            inputs=[
                self.wp_states[-1].wp_x,
                self.wp_current_object_points,
                self.wp_current_object_motions_valid,
                self.num_valid_motions,
                cfg.track_weight,
            ],
            outputs=[self.track_loss],
        )

        wp.launch(
            compute_acc_loss,
            dim=self.num_object_points,
            inputs=[
                self.wp_states[0].wp_v,
                self.wp_states[-1].wp_v,
                self.prev_acc,
                self.num_object_points,
                self.acc_count,
                cfg.acc_weight,
            ],
            outputs=[self.acc_loss],
        )

        wp.launch(
            compute_final_loss,
            dim=1,
            inputs=[self.chamfer_loss, self.track_loss, self.acc_loss],
            outputs=[self.loss],
        )

    def calculate_simple_loss(self):
        """Compute simplified smooth-L1 loss for synthetic dataset benchmarks."""
        wp.launch(
            compute_simple_loss,
            dim=self.num_object_points,
            inputs=[
                self.wp_states[-1].wp_x,
                self.wp_current_object_points,
                self.num_object_points,
            ],
            outputs=[self.loss],
        )

    def clear_loss(self):
        """Zero all loss-related Warp buffers, respecting dataset modality."""
        if cfg.data_type == "real":
            self.distance_matrix.zero_()
            self.neigh_indices.zero_()
            self.chamfer_loss.zero_()
            self.track_loss.zero_()
            self.acc_loss.zero_()
        self.loss.zero_()

    # Functions used to load the parmeters
    def set_spring_Y(self, spring_Y):
        """Copy optimised logarithmic Young's modulus into Warp parameter buffers.

        Parameters
        ----------
        spring_Y : torch.Tensor
            Tensor of length `n_springs` containing log-scale stiffness values.
        """
        # assert spring_Y.shape[0] == self.n_springs
        wp.launch(
            copy_float,
            dim=self.n_springs,
            inputs=[spring_Y],
            outputs=[self.wp_spring_Y],
        )

    def set_collide(self, collide_elas, collide_fric):
        """Update ground-plane collision parameters (restitution and friction).

        Parameters
        ----------
        collide_elas : torch.Tensor
            Scalar tensor storing restitution coefficient.
        collide_fric : torch.Tensor
            Scalar tensor storing friction coefficient.
        """
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_elas],
            outputs=[self.wp_collide_elas],
        )
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_fric],
            outputs=[self.wp_collide_fric],
        )

    def set_collide_object(self, collide_object_elas, collide_object_fric):
        """Write object-object collision parameters into Warp buffers.

        Parameters
        ----------
        collide_object_elas : torch.Tensor
            Scalar tensor containing restitution for object-object interactions.
        collide_object_fric : torch.Tensor
            Scalar tensor containing friction for object-object interactions.
        """
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_object_elas],
            outputs=[self.wp_collide_object_elas],
        )
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_object_fric],
            outputs=[self.wp_collide_object_fric],
        )
