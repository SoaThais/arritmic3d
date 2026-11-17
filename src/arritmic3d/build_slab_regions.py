import json
import os
import numpy as np


def _write_field_values(grid, field, val, write_inds, n_points, mode="overwrite"):
    """
    Create/validate the target field array in grid.point_data and write `val` at indices `write_inds`.
    - mode: 
        * "overwrite": substitui o valor existente (comportamento padrão).
        * "bitwise_or": combina bits (ex: 1 | 2 = 3), permitindo estímulos sobrepostos independentes.
        * "keep": só escreve onde o nó ainda não tem valor atribuído (valor == 0).
    """
    is_vector = isinstance(val, (list, tuple, np.ndarray)) and not isinstance(val, (str, bytes))

    if field in grid.point_data:
        current = grid.point_data[field].copy()
        if current.ndim == 1:
            if is_vector:
                raise ValueError(f"Field '{field}' exists as scalar but region provides vector value.")
            target = current
        else:
            if not is_vector:
                raise ValueError(f"Field '{field}' exists as vector but region provides scalar value.")
            comps = current.shape[1]
            if len(val) != comps:
                raise ValueError(f"Field '{field}' expects {comps} components, region provides {len(val)}.")
            target = current
    else:
        if is_vector:
            comps = len(val)
            target = np.zeros((n_points, comps), dtype=float)
        else:
            # Para bitwise_or, o tipo deve ser int
            target = np.zeros(n_points, dtype=int) if (isinstance(val, int) or mode == "bitwise_or") else np.zeros(n_points, dtype=float)

    if len(write_inds) == 0:
        return

    # Aplicação do valor conforme o modo
    if mode == "bitwise_or":
        if is_vector:
            raise ValueError("O modo 'bitwise_or' só é suportado para campos escalares inteiros.")
        target = target.astype(int)
        target[write_inds] = target[write_inds] | int(val)

    elif mode == "keep":
        if is_vector:
            mask_empty = np.all(target[write_inds] == 0, axis=-1)
        else:
            mask_empty = (target[write_inds] == 0)
        write_inds_filtered = write_inds[mask_empty]
        if len(write_inds_filtered) > 0:
            if is_vector:
                target[write_inds_filtered] = np.array(val, dtype=float)
            else:
                target[write_inds_filtered] = val

    else:  # "overwrite"
        if is_vector:
            target[write_inds] = np.array(val, dtype=float)
        else:
            target[write_inds] = val

    grid.point_data[field] = target


def set_region(grid, center, radius, data, shape="circle", z_layers=None, mode="overwrite"):
    """Write the values in `data` into grid.point_data for points inside the 2D mask.

    - center: (cx, cy) in world distance units
    - radius: scalar radius in world units (applies to the mask generation)
    - data: dict { field_name: value } where value is scalar or list-like (vector)
    - shape: "circle", "square", "diamond" or "concave" (determines mask metric)
    - z_layers: iterable de índices de fatias Z. Se None, aplica em todo o slab (0 a nz-1).
    - mode: "overwrite", "bitwise_or" ou "keep"
    """
    cx, cy = center
    x_coords = np.asarray(grid.x)
    y_coords = np.asarray(grid.y)

    # Create 2D coordinate arrays (i -> x, j -> y)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

    r = float(radius)

    if shape == "circle":
        dist2d = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    elif shape == "concave":
        dist2d = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    elif shape == "square":
        # Chebyshev distance: max(|dx|, |dy|) <= r  => square of half-side r
        dist2d = np.maximum(np.abs(X - cx), np.abs(Y - cy))
    elif shape == "diamond":
        # Manhattan distance: |dx| + |dy| <= r  => diamond shape
        dist2d = np.abs(X - cx) + np.abs(Y - cy)
    else:
        raise ValueError("Unsupported shape: choose 'circle', 'square', 'diamond' or 'concave'")

    if shape == "concave":
        mask2d = dist2d >= r
    else:
        mask2d = dist2d <= r

    i_inds, j_inds = np.nonzero(mask2d)

    nx = x_coords.size
    ny = y_coords.size
    nz = grid.dimensions[2]

    # Delimita quais camadas Z serão afetadas
    if z_layers is None:
        target_k = range(nz)
    else:
        target_k = [k for k in z_layers if 0 <= k < nz]

    # Compute flat indices
    flat_inds_all = []
    for k in target_k:
        ks = np.full_like(i_inds, k)
        flat_inds = np.ravel_multi_index((i_inds, j_inds, ks), (nx, ny, nz), order="F")
        flat_inds_all.append(flat_inds)

    if not flat_inds_all:
        return
    flat_inds_all = np.concatenate(flat_inds_all)

    n_points = grid.number_of_points
    # For each field in data, prepare target array and assign value(s)
    for field, val in data.items():
        _write_field_values(grid, field, val, flat_inds_all, n_points, mode=mode)


def load_regions_from_file(path):
    """Load regions list from a JSON file. Raise ValueError on invalid format."""
    if not os.path.isfile(path):
        raise ValueError(f"Regions file not found: {path}")
    try:
        with open(path, "r") as fh:
            data = json.load(fh)
    except Exception as e:
        raise ValueError(f"Error reading regions file '{path}': {e}")
    if not isinstance(data, list):
        raise ValueError("Regions file must contain a JSON list of region objects.")
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Regions file: element {i} is not an object.")
    return data


def parse_regions_from_cli(region_strings):
    """Parse --region occurrences (JSON strings). Raise ValueError on parse error."""
    parsed = []
    if not region_strings:
        return parsed
    for i, s in enumerate(region_strings):
        try:
            obj = json.loads(s)
        except Exception as e:
            raise ValueError(f"Error parsing --region argument #{i}: invalid JSON ({e})")
        if not isinstance(obj, dict):
            raise ValueError(f"--region argument #{i} must be a JSON object.")
        parsed.append(obj)
    return parsed


def validate_region(region, index):
    """Validate a single region dict. Raises ValueError if invalid."""
    if "shape" not in region:
        raise ValueError(f"Region {index}: missing required field 'shape'.")
    shape = str(region["shape"]).lower()

    if shape in ("circle", "square", "diamond", "concave"):
        required = ["cx", "cy", "r1", "r2"]
    elif shape == "side":
        required = ["side"]
    elif shape == "node_ids":
        required = ["ids"]
    else:
        raise ValueError(f"Region {index}: unknown shape '{region.get('shape')}'. Supported: circle,square,diamond,side,node_ids.")

    missing = [k for k in required if k not in region]
    if missing:
        raise ValueError(f"Region {index} (shape='{shape}'): missing required fields {missing}.")

    # Additional validation for side and node_ids
    if shape == "side":
        side = str(region["side"]).lower()
        if side not in {"north", "south", "east", "west"}:
            raise ValueError(f"Region {index} (shape='side'): side must be one of north,south,east,west. Got '{side}'.")
    elif shape == "node_ids":
        ids = region["ids"]
        if not isinstance(ids, (list, tuple)):
            raise ValueError(f"Region {index} (shape='node_ids'): 'ids' must be a list or tuple.")
        if len(ids) == 0:
            raise ValueError(f"Region {index} (shape='node_ids'): 'ids' cannot be empty.")
        for i, nid in enumerate(ids):
            try:
                int(nid)
            except Exception:
                raise ValueError(f"Region {index} (shape='node_ids'): ids[{i}] must be an integer.")

    if "thickness" in region:
        try:
            int(region["thickness"])
        except Exception:
            raise ValueError(f"Region {index}: 'thickness' deve ser um número inteiro.")
            
    if "mode" in region:
        valid_modes = {"overwrite", "bitwise_or", "keep"}
        if region["mode"] not in valid_modes:
            raise ValueError(f"Region {index}: 'mode' inválido ('{region['mode']}'). Escolha entre {valid_modes}.")

    known = set(required + ["shape", "thickness", "mode"])
    target_fields = [k for k in region.keys() if k not in known]
    if not target_fields:
        raise ValueError(f"Region {index}: no target fields provided (e.g. 'restitution_model' or 'activation_region').")

    # Unified validation for all shapes
    for field in target_fields:
        val = region[field]
        _validate_field_value(field, val, shape, index)


def _validate_field_value(field, val, shape, region_index):
    """
    Validate a single field value across all shape types.
    - For circle/square/diamond: allow scalar, vector, or list of values (gradient)
    - For side/node_ids: allow scalar or vector only (no gradients)
    """
    supports_gradient = shape in ("circle", "square", "diamond")

    if isinstance(val, (list, tuple)):
        if len(val) == 0:
            raise ValueError(f"Region {region_index}: field '{field}' cannot be empty.")

        # If any element is list/tuple/ndarray => vector entry (no gradient)
        contains_nested = any(isinstance(elem, (list, tuple, np.ndarray)) for elem in val)

        if supports_gradient and not contains_nested:
            # All elements are scalar: treat as gradient only if len > 1
            if len(val) == 1:
                try:
                    float(val[0])
                except Exception:
                    raise ValueError(f"Region {region_index}: field '{field}'[0] must be numeric.")
            else:
                for i, elem in enumerate(val):
                    try:
                        float(elem)
                    except Exception:
                        raise ValueError(f"Region {region_index}: field '{field}'[{i}] must be numeric.")
        else:
            try:
                for elem in val:
                    float(elem) if not isinstance(elem, (list, tuple, np.ndarray)) else None
            except Exception:
                raise ValueError(f"Region {region_index}: field '{field}' vector elements must be numeric.")
    else:
        try:
            float(val)
        except Exception:
            raise ValueError(f"Region {region_index}: field '{field}' must be numeric, vector, or list.")


def apply_region(grid, region):
    """Apply a validated region dict to the grid."""
    shape = str(region["shape"]).lower()
    known = {"shape", "cx", "cy", "r1", "r2", "side", "ids", "thickness", "mode"}
    targets = {k: region[k] for k in region.keys() if k not in known}

    thickness = region.get("thickness", 1)
    mode = region.get("mode", "overwrite")

    if shape in ("circle", "square", "diamond", "concave"):
        cx = float(region["cx"])
        cy = float(region["cy"])
        center = (cx, cy)
        r1 = float(region["r1"])
        r2 = float(region["r2"])
        z_layers = range(min(grid.dimensions[2], int(thickness))) if "thickness" in region else None
        _apply_shape_with_gradient(grid, shape, center, r1, r2, targets, z_layers=z_layers, mode=mode)

    elif shape == "side":
        side = str(region["side"]).lower()
        _apply_side_region(grid, side, targets, thickness=thickness, mode=mode)

    elif shape == "node_ids":
        ids = [int(nid) for nid in region["ids"]]
        _apply_node_ids_region(grid, ids, targets, mode=mode)


def _apply_shape_with_gradient(grid, shape, center, r1, r2, targets, z_layers=None, mode="overwrite"):
    if r1 > r2:
        raise ValueError("r1 must be less than or equal to r2.")

    for field, val in targets.items():
        if not isinstance(val, (list, tuple)) or r1 == r2:
            v = val[0] if isinstance(val, (list, tuple)) else val
            data = {field: v}
            set_region(grid, center=center, radius=r2, data=data, shape=shape, z_layers=z_layers, mode=mode)
        else:
            n_layers = len(val)
            radii = np.linspace(r1, r2, n_layers)
            for i in range(n_layers - 1, -1, -1):
                data = {field: val[i]}
                set_region(grid, center=center, radius=radii[i], data=data, shape=shape, z_layers=z_layers, mode=mode)


def _apply_side_region(grid, side, targets, thickness=1, mode="overwrite"):
    """
    Aplica valores aos nós de um dos lados do slab.
    - thickness: quantidade de camadas de nós de profundidade (default=1).
    - mode: 'overwrite', 'bitwise_or' ou 'keep'.
    """
    if side not in {"north", "south", "east", "west"}:
        raise ValueError("side must be one of: north, south, east, west")

    dims = grid.dimensions
    nx, ny, nz = dims
    thickness = max(1, int(thickness))

    if side == "north":
        y_indices = list(range(max(0, ny - thickness), ny))
        x_indices = list(range(nx))
    elif side == "south":
        y_indices = list(range(min(thickness, ny)))
        x_indices = list(range(nx))
    elif side == "east":
        x_indices = list(range(max(0, nx - thickness), nx))
        y_indices = list(range(ny))
    else:  # west
        x_indices = list(range(min(thickness, nx)))
        y_indices = list(range(ny))

    z_indices = list(range(nz))

    flat_inds_all = []
    for z in z_indices:
        for y in y_indices:
            for x in x_indices:
                idx = np.ravel_multi_index((x, y, z), dims, order="F")
                flat_inds_all.append(idx)
    flat_inds_all = np.array(flat_inds_all)

    if len(flat_inds_all) == 0:
        return

    n_points = grid.number_of_points
    for field, val in targets.items():
        _write_field_values(grid, field, val, flat_inds_all, n_points, mode=mode)


def _apply_node_ids_region(grid, ids, targets, mode="overwrite"):
    """
    Apply field values to explicitly specified node IDs.
    """
    if len(ids) == 0:
        return

    n_points = grid.number_of_points
    write_inds = np.array(ids)
    for field, val in targets.items():
        _write_field_values(grid, field, val, write_inds, n_points, mode=mode)