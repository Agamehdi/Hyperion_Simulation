import streamlit as st
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import pandas as pd
import io
import re
import hashlib


# ----------------------------
# Distribution sampling helpers (SciPy-free)
# ----------------------------
DIST_OPTIONS = ["Uniform", "Triangular", "Normal"]

def sample_range(rng, dist="Uniform", size=None):
    """Sample values between [low, high] using chosen distribution.
    - Uniform: U(low, high)
    - Triangular: triangular(low, mode, high) where mode=(low+high)/2
    - Normal: N(mu, sigma) clipped to [low, high] with mu=(low+high)/2 and sigma=(high-low)/6
    """
    low, high = float(rng[0]), float(rng[1])
    if high < low:
        low, high = high, low
    if dist == "Triangular":
        mode = (low + high) / 2.0
        return np.random.triangular(low, mode, high, size=size)
    if dist == "Normal":
        mu = (low + high) / 2.0
        sigma = max((high - low) / 6.0, 1e-9)
        x = np.random.normal(mu, sigma, size=size)
        return np.clip(x, low, high)
    # default Uniform
    return np.random.uniform(low, high, size=size)

def get_dist(key: str, default: str = "Uniform") -> str:
    """Read distribution selection from session_state (default Uniform)."""
    return st.session_state.get(key, default)


def _dist_select(label, key, default="Uniform", container=None):
    if container is None:
        container = st
    return container.selectbox(label, DIST_OPTIONS, index=DIST_OPTIONS.index(default), key=key)

# ----------------------------
# Input mode helpers (Slicers vs Manual entry)
# ----------------------------
def _sidebar_value(label, min_value, max_value, default, step=None, key=None, help=None, fmt=None):
    """Single value input: slider (Slicers) or number_input (Manual entry)."""
    mode = st.session_state.get("input_mode", "Slicers")
    if mode == "Manual entry":
        # number_input expects step not None; infer a reasonable one
        if step is None:
            step = 1 if isinstance(default, int) and isinstance(min_value, int) and isinstance(max_value, int) else 0.01
        return st.sidebar.number_input(label, min_value=min_value, max_value=max_value, value=default, step=step, key=key, help=help, format=fmt)
    else:
        kwargs = {}
        if step is not None:
            kwargs["step"] = step
        if fmt is not None:
            kwargs["format"] = fmt
        return st.sidebar.slider(label, min_value, max_value, default, key=key, help=help, **kwargs)

def _sidebar_range(label, min_value, max_value, default, step=None, key=None, help=None, fmt=None):
    """Range input: range slider (Slicers) or two number_inputs (Manual entry)."""
    mode = st.session_state.get("input_mode", "Slicers")
    if mode == "Manual entry":
        if key is None:
            key = re.sub(r'\W+', '_', label.lower()).strip('_')
        # infer steps
        if step is None:
            step = 1 if all(isinstance(x, int) for x in [default[0], default[1], min_value, max_value]) else 0.01
        c1, c2 = st.sidebar.columns(2)
        with c1:
            vmin = st.number_input(f"{label} (min)", min_value=min_value, max_value=max_value, value=default[0], step=step, key=f"{key}_min", help=help, format=fmt)
        with c2:
            vmax = st.number_input(f"{label} (max)", min_value=min_value, max_value=max_value, value=default[1], step=step, key=f"{key}_max", help=help, format=fmt)
        if vmin > vmax:
            st.sidebar.error(f"{label}: min cannot be greater than max.")
        return (vmin, vmax)
    else:
        kwargs = {}
        if step is not None:
            kwargs["step"] = step
        if fmt is not None:
            kwargs["format"] = fmt
        return st.sidebar.slider(label, min_value, max_value, default, key=key, help=help, **kwargs)
# Initialize session state for project management / run gating
if "project_created" not in st.session_state:
    st.session_state.project_created = False
if "project_name" not in st.session_state:
    st.session_state.project_name = ""
if "parameters" not in st.session_state:
    st.session_state.parameters = {}
if "last_run_hash" not in st.session_state:
    st.session_state.last_run_hash = None


# ----------------------------
# JSON-safe serialization and run gating helpers
# ----------------------------
def _json_safe(obj):
    """Convert numpy / pandas / tuples into JSON-serializable plain python types."""
    if obj is None:
        return None
    # numpy scalars
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    # numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # pandas
    if hasattr(obj, "to_pydatetime"):
        try:
            return obj.to_pydatetime().isoformat()
        except Exception:
            pass
    if isinstance(obj, (datetime,)):
        return obj.isoformat(sep=" ")
    # containers
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    # fallback
    return obj

def _param_hash(params: dict) -> str:
    """Stable hash of parameters dict for detecting changes."""
    safe = _json_safe(params)
    blob = json.dumps(safe, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

parameters = {}

# Sidebar Project Management
st.sidebar.title("Project Management")
project_option = st.sidebar.radio("Project Option", ["Create New Project", "Load Existing Project"])

# Create New Project
if project_option == "Create New Project":
    project_name = st.sidebar.text_input("Enter Project Name")
    if st.sidebar.button("Create Project"):
        if project_name:
            st.session_state.project_created = True
            st.session_state.project_name = project_name
            st.session_state.parameters = {}
            st.success(f"Project '{project_name}' created successfully!")
        else:
            st.error("Please enter a valid project name.")

# Load Existing Project
elif project_option == "Load Existing Project":
    uploaded_file = st.sidebar.file_uploader("Upload Project File (JSON)", type=["json"])
    if uploaded_file is not None:
        project_data = json.load(uploaded_file)
        st.session_state.project_created = True
        st.session_state.project_name = project_data.get("project_name", "Loaded Project")
        st.session_state.parameters = project_data.get("parameters", {})
        # Restore saved UI choices before the related widgets are instantiated.
        for k, v in st.session_state.parameters.items():
            if k.startswith("dist_") or k == "input_mode":
                st.session_state[k] = v

        st.success(f"Project '{st.session_state.project_name}' loaded successfully!")

if st.session_state.project_created:
    st.sidebar.title(f"Project: {st.session_state.project_name}")
    st.title("Hyperion Simulation Model")

    # Field Type Selection
    field_type_options = [
        "Oil Field",
        "Gas Field",
        "Oil Field with WI",
        "Oil Field with WI & GI",
    ]
    saved_field_type = st.session_state.parameters.get("field_type", "Oil Field")
    if saved_field_type not in field_type_options:
        saved_field_type = "Oil Field"

    field_type = st.sidebar.radio(
        "Select Field Type:",
        field_type_options,
        index=field_type_options.index(saved_field_type),
    )
    is_oil_field = field_type != "Gas Field"



    # Input mode (keep sliders vs manual entry)
    st.sidebar.subheader("Input Mode")
    input_mode = st.sidebar.radio(
        "How do you want to enter parameters?",
        ["Slicers", "Manual entry"],
        index=0 if st.session_state.get("input_mode", "Slicers") == "Slicers" else 1,
        help="Slicers = sliders/range sliders. Manual entry = replace sliders with numeric entry boxes."
    )
    st.session_state.input_mode = input_mode

# ----------------------------
# ----------------------------
# Right-side panel: Parameter Distribution (default Uniform)
# ----------------------------
# Distribution Option toggle (no tabs)
# ----------------------------
if st.session_state.project_created:
    # Choose which panel to show in the LEFT sidebar
    st.sidebar.markdown("---")
    _panel = st.sidebar.radio(
        "Panel",
        ["Main", "Distribution Option"],
        index=0,
        key="panel_mode"
    )

    # Defaults BEFORE widgets are instantiated (so they exist even if user never opens Distribution Option)
    for _k in [
        "dist_initial_pressure","dist_stoiip","dist_ogiip","dist_aquifer",
        "dist_initial_rate","dist_incremental","dist_decline","dist_arpsb",
        "dist_wi_rate","dist_vow","dist_gi_rate","dist_vog"
    ]:
        st.session_state.setdefault(_k, "Uniform")

    if _panel == "Distribution Option":
        st.sidebar.markdown("### Parameter Distribution")
        st.sidebar.caption("Default is **Uniform**. Change only if you need.")
        _show_wi = field_type in ["Oil Field with WI", "Oil Field with WI & GI"]
        _show_gi = field_type in ["Oil Field with WI & GI"]

        # Core ranged parameters
        _dist_select("Initial Pressure", "dist_initial_pressure", container=st.sidebar)
        if field_type == "Gas Field":
            _dist_select("OGIIP", "dist_ogiip", container=st.sidebar)
        else:
            _dist_select("STOIIP", "dist_stoiip", container=st.sidebar)

        _dist_select("Initial Rate", "dist_initial_rate", container=st.sidebar)
        _dist_select("Incremental", "dist_incremental", container=st.sidebar)
        _dist_select("Decline", "dist_decline", container=st.sidebar)
        _dist_select("Arps b", "dist_arpsb", container=st.sidebar)

        if _show_wi:
            st.sidebar.markdown("---")
            st.sidebar.markdown("**Water Injection**")
            _dist_select("Aquifer Strength", "dist_aquifer", container=st.sidebar)
            _dist_select("WI Rate", "dist_wi_rate", container=st.sidebar)
            _dist_select("Value of Water", "dist_vow", container=st.sidebar)

        if _show_gi:
            st.sidebar.markdown("---")
            st.sidebar.markdown("**Gas Injection**")
            _dist_select("GI Rate", "dist_gi_rate", container=st.sidebar)
            _dist_select("Value of Gas", "dist_vog", container=st.sidebar)

    _col_main = st.container()
    with _col_main:
        # Main UI continues below as-is
        pass

    # Convenience locals used later (e.g., project save)
    dist_initial_pressure = st.session_state.get("dist_initial_pressure", "Uniform")
    dist_stoiip = st.session_state.get("dist_stoiip", "Uniform")
    dist_ogiip = st.session_state.get("dist_ogiip", "Uniform")
    dist_aquifer = st.session_state.get("dist_aquifer", "Uniform")
    dist_initial_rate = st.session_state.get("dist_initial_rate", "Uniform")
    dist_incremental = st.session_state.get("dist_incremental", "Uniform")
    dist_decline = st.session_state.get("dist_decline", "Uniform")
    dist_arpsb = st.session_state.get("dist_arpsb", "Uniform")
    dist_wi_rate = st.session_state.get("dist_wi_rate", "Uniform")
    dist_vow = st.session_state.get("dist_vow", "Uniform")
    dist_gi_rate = st.session_state.get("dist_gi_rate", "Uniform")
    dist_vog = st.session_state.get("dist_vog", "Uniform")
# General Parameters
    project_start_date = st.sidebar.date_input("Project Start Date", datetime.strptime(st.session_state.parameters.get("project_start_date", "2027-01-01"), "%Y-%m-%d"))
    project_end_date = st.sidebar.date_input("Project End Date", datetime.strptime(st.session_state.parameters.get("project_end_date", "2035-01-01"), "%Y-%m-%d"))
    max_slots = _sidebar_value("Max Well Slots", 10, 1000, st.session_state.parameters.get("max_slots", 500), step=1, key="max_slots")
    # max_slots = _sidebar_value("Max Well Slots", 10, 1000, st.session_state.parameters.get("max_slots", 500), step=1, key="max_slots")

    # Well slot strategy: Option A (kill oldest) vs Option B (keep highest-rate wells)
    if project_end_date <= project_start_date:
        st.error("Project End Date must be later than Project Start Date.")
        st.stop()

    slot_options = ["Option A – kill oldest wells", "Option B – keep top-rate wells"]
    saved_slot_strategy = st.session_state.parameters.get("slot_strategy", slot_options[0])
    if saved_slot_strategy not in slot_options:
        saved_slot_strategy = slot_options[0]
    slot_strategy = st.sidebar.radio(
        "When max slots are full:",
        slot_options,
        index=slot_options.index(saved_slot_strategy),
        help="Option A: drop the oldest well first. Option B: always keep the highest-rate wells alive.",
    )


    
    initial_pressure_range = _sidebar_range("Initial Reservoir Pressure (psia)", 1000, 10000, tuple(st.session_state.parameters.get("initial_pressure_range", (3200, 4200))), step=1, key="initial_pressure_range")

    # manual_schedule = []
    # drilling_duration = []
    # oil_color = "lightgreen"
    # gas_color="salmon"
    # water_color="lightblue"
    
    # Oil/Gas Field Parameters
    if field_type == "Oil Field":
        st.write("**Oil Field Selected**")
        profile_color = "springgreen"  # Light blue for oil field
        rate_label = "Oil Rate (Mstb/d)"
        cum_label = "Cumulative Oil (MMSTB)"
        incr_label = "Incremental Oil (MMSTB)"
        title_suffix = "Oil"
        maximum_rate = _sidebar_value("Maximum Oil Rate Limit (Mstb/d)", 10.0, 200.0, st.session_state.parameters.get("maximum_rate", 100.0), step=0.1, key="maximum_rate")
        stoiip_range = _sidebar_range("STOIIP (MMSTB)", 100, 10000, tuple(st.session_state.parameters.get("stoiip_range", (500, 1000))), step=1, key="stoiip_range")
        residual_oil_fraction = _sidebar_value("Residual Oil Fraction", 0.0, 0.5, st.session_state.parameters.get("residual_oil_fraction", 0.2), step=0.01, key="residual_oil_fraction")
        degradation_factor = _sidebar_value("Well Degradation Factor", 0.1, 1.0, st.session_state.parameters.get("degradation_factor", 0.98), step=0.001, key="degradation_factor")
        adjust_schedule = st.sidebar.checkbox("Adjust Drilling Schedule", value=st.session_state.parameters.get("adjust_schedule", False))
        manual_schedule = st.session_state.parameters.get("manual_schedule", []) or []
        drilling_duration = st.session_state.parameters.get("drilling_duration", (85, 120)) or (85, 120)

        if adjust_schedule and manual_schedule:
            updated_manual_schedule = []
            for year, wells in manual_schedule:
                edited_wells = st.sidebar.number_input(
                    f"Year {year} - Number of Wells",
                    min_value=0,
                    value=int(wells),
                    step=1,
                    key=f"year_{year}_wells",
                )
                updated_manual_schedule.append((int(year), int(edited_wells)))
            manual_schedule = updated_manual_schedule
        elif adjust_schedule:
            num_years = st.sidebar.number_input("Number of Years for Drilling Schedule", min_value=1, value=5, step=1)
            manual_schedule = []
            for year in range(1, num_years + 1):
                num_wells = st.sidebar.number_input(f"Year {year} - Number of Wells", min_value=0, value=5, step=1, key=f"year_{year}_wells")
                manual_schedule.append((year, num_wells))
        else:
            num_producers = _sidebar_range("Number of Producers", 1, 100, tuple(st.session_state.parameters.get("num_producers", (10, 20)) or (14, 15)), step=1, key="num_producers")
            drilling_duration = _sidebar_range("Drilling Duration (days)", 0, 200, tuple(st.session_state.parameters.get("drilling_duration", (60, 90)) or (85, 120)), step=1, key="drilling_duration")
    
        # Validation for `num_producers`
        if not adjust_schedule and num_producers[0] >= num_producers[1]:
            st.error("Please select at least 2 wells.")
            st.stop()
        
    elif field_type == "Gas Field":
        st.write("**Gas Field Selected**")
        profile_color = "#ffcccb"  # Light red for gas field
        rate_label = "Gas Rate (MMscf/d)"
        cum_label = "Cumulative Gas (Bcf)"
        incr_label = "Incremental Gas (Bcf)"
        title_suffix = "Gas"
        maximum_rate = _sidebar_value("Maximum Gas Rate Limit (MMscf/d)", 10.0, 200.0, st.session_state.parameters.get("maximum_rate", 100.0), step=0.1, key="maximum_rate")
        ogiip_range = _sidebar_range("OGIIP (Bcf)", 100, 10000, tuple(st.session_state.parameters.get("ogiip_range", (500, 1000))), step=1, key="ogiip_range")
        gas_recovery_factor = _sidebar_value("Gas Recovery Factor", 0.1, 0.9, st.session_state.parameters.get("gas_recovery_factor", 0.7), step=0.01, key="gas_recovery_factor")
        degradation_factor = _sidebar_value("Well Degradation Factor", 0.1, 1.0, st.session_state.parameters.get("degradation_factor", 0.98), step=0.001, key="degradation_factor")
        adjust_schedule = st.sidebar.checkbox("Adjust Drilling Schedule", value=st.session_state.parameters.get("adjust_schedule", False))
        manual_schedule = st.session_state.parameters.get("manual_schedule", [])
        drilling_duration = st.session_state.parameters.get("drilling_duration", (85, 120))  # Default to saved value or a sensible range

        if adjust_schedule and manual_schedule:
            # Populate and update manual schedule from a loaded project
            updated_manual_schedule = []
            for year, wells in manual_schedule:
                edited_wells = st.sidebar.number_input(
                    f"Year {year} - Number of Wells",
                    min_value=0,
                    value=int(wells),
                    step=1,
                    key=f"year_{year}_wells",
                )
                updated_manual_schedule.append((int(year), int(edited_wells)))
            manual_schedule = updated_manual_schedule
        elif adjust_schedule:
            # Initialize manual schedule if not already defined
            num_years = st.sidebar.number_input("Number of Years for Drilling Schedule", min_value=1, value=5, step=1)
            manual_schedule = []
            for year in range(1, num_years + 1):
                num_wells = st.sidebar.number_input(f"Year {year} - Number of Wells", min_value=0, value=5, step=1, key=f"year_{year}_wells")
                manual_schedule.append((year, num_wells))
        else:
            num_producers = _sidebar_range("Number of Producers", 1, 100, tuple(st.session_state.parameters.get("num_producers", (21, 48)) or (21, 48)), step=1, key="num_producers")
            drilling_duration = _sidebar_range("Drilling Duration (days)", 0, 200, tuple(st.session_state.parameters.get("drilling_duration", (85, 120)) or (85, 120)), step=1, key="drilling_duration")

        
    elif field_type == "Oil Field with WI":
        st.write("**Oil Field with Water Injection Selected**")
        profile_color = "springgreen"
        rate_label = "Oil Rate (Mstb/d)"
        cum_label = "Cumulative Oil (MMSTB)"
        incr_label = "Incremental Oil (MMSTB)"
        title_suffix = "Oil"
        maximum_rate = _sidebar_value("Maximum Oil Rate Limit (Mstb/d)", 10.0, 200.0, st.session_state.parameters.get("maximum_rate", 100.0), step=0.1, key="maximum_rate")
        max_water_injection_rate = _sidebar_value("Maximum WI Rate Limit (Mstb/d)", 10.0, 200.0, st.session_state.parameters.get("maximum_win_rate", 100.0), step=0.1, key="maximum_win_rate")
        stoiip_range = _sidebar_range("STOIIP (MMSTB)", 100, 10000, tuple(st.session_state.parameters.get("stoiip_range", (500, 1000))), step=1, key="stoiip_range")
        residual_oil_fraction = _sidebar_value("Residual Oil Fraction", 0.0, 0.5, st.session_state.parameters.get("residual_oil_fraction", 0.2), step=0.01, key="residual_oil_fraction")
        degradation_factor = _sidebar_value("Well Degradation Factor", 0.1, 1.0, st.session_state.parameters.get("degradation_factor", 0.98), step=0.001, key="degradation_factor")

        # WI mode currently requires explicit producer and injector schedules.
        adjust_schedule = False
        drilling_duration = None
        num_producers = None
        manual_schedule = []

        manual_producer_schedule = st.session_state.parameters.get("manual_producer_schedule", []) or []
        manual_injector_schedule = st.session_state.parameters.get("manual_injector_schedule", []) or []

        adjust_producer_schedule = st.sidebar.checkbox(
            "Adjust Drilling Schedule for Producers",
            value=st.session_state.parameters.get("adjust_producer_schedule", True),
        )
        adjust_injector_schedule = st.sidebar.checkbox(
            "Adjust Drilling Schedule for Water Injectors",
            value=st.session_state.parameters.get("adjust_injector_schedule", True),
        )

        if not (adjust_producer_schedule and adjust_injector_schedule):
            st.sidebar.warning("Please select drilling schedule for both Producers and Water Injectors to proceed.")
            st.stop()

        if manual_producer_schedule:
            updated_schedule = []
            for year, wells in manual_producer_schedule:
                edited_wells = st.sidebar.number_input(
                    f"Year {year} - Number of Producers",
                    min_value=0,
                    value=int(wells),
                    step=1,
                    key=f"producer_year_{year}_wells",
                )
                updated_schedule.append((int(year), int(edited_wells)))
            manual_producer_schedule = updated_schedule
        else:
            num_years_producers = st.sidebar.number_input("Number of Years for Producers", min_value=1, value=5, step=1)
            manual_producer_schedule = []
            for year in range(1, num_years_producers + 1):
                num_wells = st.sidebar.number_input(
                    f"Year {year} - Number of Producers",
                    min_value=0,
                    value=5,
                    step=1,
                    key=f"producer_year_{year}_wells",
                )
                manual_producer_schedule.append((year, int(num_wells)))

        if manual_injector_schedule:
            updated_schedule = []
            for year, wells in manual_injector_schedule:
                edited_wells = st.sidebar.number_input(
                    f"Year {year} - Number of Water Injectors",
                    min_value=0,
                    value=int(wells),
                    step=1,
                    key=f"injector_year_{year}_wells",
                )
                updated_schedule.append((int(year), int(edited_wells)))
            manual_injector_schedule = updated_schedule
        else:
            num_years_injectors = st.sidebar.number_input("Number of Years for Water Injectors", min_value=1, value=3, step=1)
            manual_injector_schedule = []
            for year in range(1, num_years_injectors + 1):
                num_injectors = st.sidebar.number_input(
                    f"Year {year} - Number of Water Injectors",
                    min_value=0,
                    value=3,
                    step=1,
                    key=f"injector_year_{year}_wells",
                )
                manual_injector_schedule.append((year, int(num_injectors)))

        if sum(wells for _, wells in manual_producer_schedule) <= 0:
            st.error("Producer drilling schedule must contain at least one well.")
            st.stop()
        if sum(wells for _, wells in manual_injector_schedule) <= 0:
            st.error("Water-injector drilling schedule must contain at least one well.")
            st.stop()

        water_injection_rate = _sidebar_range("Water Injection Rate (Mstb/d)", 0.1, 20.0, tuple(st.session_state.parameters.get("water_injection_rate", (1.0, 10.0))), step=0.1, key="water_injection_rate")
        value_of_water = _sidebar_range("Value of Water", 0.0, 1.0, tuple(st.session_state.parameters.get("value_of_water", (0.3, 0.5))), step=0.01, key="value_of_water")
        aquifer_strength_range = _sidebar_range("Aquifer Strength Range", 0.0, 1.0, tuple(st.session_state.parameters.get("aquifer_strength_range", (0.3, 0.5))), step=0.01, key="aquifer_strength_range")
        water_injection_well_life = _sidebar_range("Water Injection Well Life (years)", 1, 50, tuple(st.session_state.parameters.get("water_injection_well_life", (20, 25))), step=1, key="water_injection_well_life")
        water_injection_efficiency = _sidebar_value("Water Injection Efficiency", 0.0, 1.0, st.session_state.parameters.get("water_injection_efficiency", 0.5), step=0.01, key="water_injection_efficiency")

    elif field_type == "Oil Field with WI & GI":
        st.write("**Oil Field with Water and Gas Injection Selected**")
        profile_color = "springgreen"
        rate_label = "Oil Rate (Mstb/d)"
        cum_label = "Cumulative Oil (MMSTB)"
        incr_label = "Incremental Oil (MMSTB)"
        title_suffix = "Oil"
        maximum_rate = _sidebar_value("Maximum Oil Rate Limit (Mstb/d)", 10.0, 200.0, st.session_state.parameters.get("maximum_rate", 100.0), step=0.1, key="maximum_rate")
        max_water_injection_rate = _sidebar_value("Maximum WI Rate Limit (Mstb/d)", 10.0, 200.0, st.session_state.parameters.get("maximum_win_rate", 100.0), step=0.1, key="maximum_win_rate")
        max_gas_injection_rate = _sidebar_value("Maximum GI Rate Limit (MMscf/d)", 10.0, 250.0, st.session_state.parameters.get("maximum_gin_rate", 100.0), step=0.1, key="maximum_gin_rate")
        stoiip_range = _sidebar_range("STOIIP (MMSTB)", 100, 10000, tuple(st.session_state.parameters.get("stoiip_range", (500, 1000))), step=1, key="stoiip_range")
        residual_oil_fraction = _sidebar_value("Residual Oil Fraction", 0.0, 0.5, st.session_state.parameters.get("residual_oil_fraction", 0.2), step=0.01, key="residual_oil_fraction")
        degradation_factor = _sidebar_value("Well Degradation Factor", 0.1, 1.0, st.session_state.parameters.get("degradation_factor", 0.98), step=0.001, key="degradation_factor")

        # WI & GI mode currently requires explicit schedules for all well types.
        adjust_schedule = False
        drilling_duration = None
        num_producers = None
        manual_schedule = []

        manual_producer_schedule = st.session_state.parameters.get("manual_producer_schedule", []) or []
        manual_injector_schedule = st.session_state.parameters.get("manual_injector_schedule", []) or []
        manual_gas_injector_schedule = st.session_state.parameters.get("manual_gas_injector_schedule", []) or []

        adjust_producer_schedule = st.sidebar.checkbox(
            "Adjust Drilling Schedule for Producers",
            value=st.session_state.parameters.get("adjust_producer_schedule", True),
        )
        adjust_water_injector_schedule = st.sidebar.checkbox(
            "Adjust Drilling Schedule for Water Injectors",
            value=st.session_state.parameters.get("adjust_water_injector_schedule", True),
        )
        adjust_gas_injector_schedule = st.sidebar.checkbox(
            "Adjust Drilling Schedule for Gas Injectors",
            value=st.session_state.parameters.get("adjust_gas_injector_schedule", True),
        )

        if not (adjust_producer_schedule and adjust_water_injector_schedule and adjust_gas_injector_schedule):
            st.sidebar.warning("Please select drilling schedule for Producers, Water Injectors, and Gas Injectors to proceed.")
            st.stop()

        if manual_producer_schedule:
            updated_schedule = []
            for year, wells in manual_producer_schedule:
                edited_wells = st.sidebar.number_input(
                    f"Year {year} - Number of Producers",
                    min_value=0,
                    value=int(wells),
                    step=1,
                    key=f"producer_year_{year}_wells",
                )
                updated_schedule.append((int(year), int(edited_wells)))
            manual_producer_schedule = updated_schedule
        else:
            num_years_producers = st.sidebar.number_input("Number of Years for Producers", min_value=1, value=5, step=1)
            manual_producer_schedule = []
            for year in range(1, num_years_producers + 1):
                num_wells = st.sidebar.number_input(
                    f"Year {year} - Number of Producers",
                    min_value=0,
                    value=5,
                    step=1,
                    key=f"producer_year_{year}_wells",
                )
                manual_producer_schedule.append((year, int(num_wells)))

        if manual_injector_schedule:
            updated_schedule = []
            for year, wells in manual_injector_schedule:
                edited_wells = st.sidebar.number_input(
                    f"Year {year} - Number of Water Injectors",
                    min_value=0,
                    value=int(wells),
                    step=1,
                    key=f"injector_year_{year}_wells",
                )
                updated_schedule.append((int(year), int(edited_wells)))
            manual_injector_schedule = updated_schedule
        else:
            num_years_water_injectors = st.sidebar.number_input("Number of Years for Water Injectors", min_value=1, value=3, step=1)
            manual_injector_schedule = []
            for year in range(1, num_years_water_injectors + 1):
                num_injectors = st.sidebar.number_input(
                    f"Year {year} - Number of Water Injectors",
                    min_value=0,
                    value=3,
                    step=1,
                    key=f"injector_year_{year}_wells",
                )
                manual_injector_schedule.append((year, int(num_injectors)))

        if manual_gas_injector_schedule:
            updated_schedule = []
            for year, wells in manual_gas_injector_schedule:
                edited_wells = st.sidebar.number_input(
                    f"Year {year} - Number of Gas Injectors",
                    min_value=0,
                    value=int(wells),
                    step=1,
                    key=f"gas_injector_year_{year}_wells",
                )
                updated_schedule.append((int(year), int(edited_wells)))
            manual_gas_injector_schedule = updated_schedule
        else:
            num_years_gas_injectors = st.sidebar.number_input("Number of Years for Gas Injectors", min_value=1, value=3, step=1)
            manual_gas_injector_schedule = []
            for year in range(1, num_years_gas_injectors + 1):
                num_injectors = st.sidebar.number_input(
                    f"Year {year} - Number of Gas Injectors",
                    min_value=0,
                    value=3,
                    step=1,
                    key=f"gas_injector_year_{year}_wells",
                )
                manual_gas_injector_schedule.append((year, int(num_injectors)))

        if sum(wells for _, wells in manual_producer_schedule) <= 0:
            st.error("Producer drilling schedule must contain at least one well.")
            st.stop()
        if sum(wells for _, wells in manual_injector_schedule) <= 0:
            st.error("Water-injector drilling schedule must contain at least one well.")
            st.stop()
        if sum(wells for _, wells in manual_gas_injector_schedule) <= 0:
            st.error("Gas-injector drilling schedule must contain at least one well.")
            st.stop()

        water_injection_rate = _sidebar_range("Water Injection Rate (Mstb/d)", 0.1, 20.0, tuple(st.session_state.parameters.get("water_injection_rate", (1.0, 10.0))), step=0.1, key="water_injection_rate")
        value_of_water = _sidebar_range("Value of Water", 0.0, 1.0, tuple(st.session_state.parameters.get("value_of_water", (0.3, 0.5))), step=0.01, key="value_of_water")
        aquifer_strength_range = _sidebar_range("Aquifer Strength Range", 0.0, 1.0, tuple(st.session_state.parameters.get("aquifer_strength_range", (0.3, 0.5))), step=0.01, key="aquifer_strength_range")
        water_injection_well_life = _sidebar_range("Water Injection Well Life (years)", 1, 50, tuple(st.session_state.parameters.get("water_injection_well_life", (5, 15))), step=1, key="water_injection_well_life")
        water_injection_efficiency = _sidebar_value("Water Injection Efficiency", 0.0, 1.0, st.session_state.parameters.get("water_injection_efficiency", 0.5), step=0.01, key="water_injection_efficiency")

        gas_injection_rate = _sidebar_range("Gas Injection Rate (MMscf/d)", 0.1, 200.0, tuple(st.session_state.parameters.get("gas_injection_rate", (10.0, 50.0))), step=0.1, key="gas_injection_rate")
        value_of_gas = _sidebar_range("Value of Gas", 0.0, 1.0, tuple(st.session_state.parameters.get("value_of_gas", (0.3, 0.5))), step=0.01, key="value_of_gas")
        gas_injection_well_life = _sidebar_range("Gas Injection Well Life (years)", 1, 50, tuple(st.session_state.parameters.get("gas_injection_well_life", (5, 15))), step=1, key="gas_injection_well_life")
        gas_injection_efficiency = _sidebar_value("Gas Injection Efficiency", 0.0, 1.0, st.session_state.parameters.get("gas_injection_efficiency", 0.5), step=0.01, key="gas_injection_efficiency")

    # st.write("Additional parameters and configurations loaded based on the selected field type.")



    # Decline Parameters
    if field_type == "Gas Field":
        initial_rate_label = "Initial Gas Rate (MMscf/d)"
        initial_rate_default = tuple(st.session_state.parameters.get("initial_rate", (10.0, 50.0)))
        initial_rate = _sidebar_range(
            initial_rate_label, 0.1, 200.0, initial_rate_default,
            step=0.1, key="initial_rate"
        )
    else:
        initial_rate_label = "Initial Oil Rate (Mstb/d)"
        initial_rate_default = tuple(st.session_state.parameters.get("initial_rate", (0.8, 3.2)))
        initial_rate = _sidebar_range(
            initial_rate_label, 0.01, 20.0, initial_rate_default,
            step=0.01, key="initial_rate"
        )
    incremental_factor = _sidebar_range("Incremental Factor", 0.0, 1.0, tuple(st.session_state.parameters.get("incremental_factor", (0.3, 0.9))), step=0.01, key="incremental_factor")
    yearly_decline = _sidebar_range("Yearly Decline (%)", 0, 200, tuple(st.session_state.parameters.get("yearly_decline", (10, 45))), step=1, key="yearly_decline")
    arps_b = _sidebar_range("Arps Decline Parameter b", 0.0, 1.0, tuple(st.session_state.parameters.get("arps_b", (0.0, 0.5))), step=0.01, key="arps_b")
    simulation_number = _sidebar_value("Number of Simulations", 100, 10000, st.session_state.parameters.get("simulation_number", 1000), step=100, key="simulation_number")





    # Economics Checkbox (always visible)
    st.sidebar.subheader("Additional Options")
    calculate_tornado = st.sidebar.checkbox("Calculate OAT (Tornado)", value=st.session_state.parameters.get("calculate_tornado", False))
    diversify_assumptions = st.sidebar.checkbox("Diversify All Wells", value=st.session_state.parameters.get("diversify_assumptions", False))
    calculate_water_cut = st.sidebar.checkbox("Calculate Water Cut", value=st.session_state.parameters.get("calculate_water_cut", False)) if is_oil_field else False
    calculate_gor = st.sidebar.checkbox("Calculate GOR", value=st.session_state.parameters.get("calculate_gor", False)) if is_oil_field else False
    calculate_cgr = st.sidebar.checkbox("Calculate CGR", value=st.session_state.parameters.get("calculate_cgr", False)) if field_type == "Gas Field" else False
    calculate_wgr = st.sidebar.checkbox("Calculate WGR", value=st.session_state.parameters.get("calculate_wgr", False)) if field_type == "Gas Field" else False
    calculate_economics = st.sidebar.checkbox("Calculate Economics", value=st.session_state.parameters.get("calculate_economics", False))
    export_csvs = st.sidebar.checkbox("Export Results to CSV", value=st.session_state.parameters.get("export_csvs", False))

    uploaded_csv = None
    uploaded_gas_type_curve = None

    if is_oil_field and (calculate_water_cut or calculate_gor):
        uploaded_csv = st.sidebar.file_uploader(
            "Upload Oil Type Curve CSV",
            type=["csv"],
            help="Required columns depend on the selected calculations: Normalized Cum Oil, MMstb; Water Cut, fr; GOR, scf/stb.",
        )

    if field_type == "Gas Field" and (calculate_cgr or calculate_wgr):
        uploaded_gas_type_curve = st.sidebar.file_uploader(
            "Upload Gas Type Curve CSV",
            type=["csv"],
            help="Ensure the CSV contains columns: 'Normalized Cumulative Gas (Bcf)', 'CGR (bbl/MMscf)', 'WGR (bbl/MMscf)'.",
        )

    # Save Parameters
    params_update = {
        "field_type": field_type,
        "project_start_date": project_start_date.isoformat(),
        "project_end_date": project_end_date.isoformat(),
        "input_mode": st.session_state.get("input_mode", "Slicers"),
        "max_slots": max_slots,
        "slot_strategy": slot_strategy,
        "initial_pressure_range": initial_pressure_range,
        "maximum_rate": maximum_rate,
        "degradation_factor": degradation_factor,
        "initial_rate": initial_rate,
        "incremental_factor": incremental_factor,
        "yearly_decline": yearly_decline,
        "arps_b": arps_b,
        "simulation_number": simulation_number,
        "dist_initial_pressure": dist_initial_pressure,
        "dist_stoiip": dist_stoiip,
        "dist_ogiip": dist_ogiip,
        "dist_aquifer": dist_aquifer,
        "dist_initial_rate": dist_initial_rate,
        "dist_incremental": dist_incremental,
        "dist_decline": dist_decline,
        "dist_arpsb": dist_arpsb,
        "dist_wi_rate": dist_wi_rate,
        "dist_vow": dist_vow,
        "dist_gi_rate": dist_gi_rate,
        "dist_vog": dist_vog,
        "calculate_tornado": calculate_tornado,
        "diversify_assumptions": diversify_assumptions,
        "calculate_water_cut": calculate_water_cut,
        "calculate_gor": calculate_gor,
        "calculate_cgr": calculate_cgr,
        "calculate_wgr": calculate_wgr,
        "calculate_economics": calculate_economics,
        "export_csvs": export_csvs,
    }

    if field_type in ("Oil Field", "Gas Field"):
        params_update.update({
            "adjust_schedule": adjust_schedule,
            "manual_schedule": manual_schedule if adjust_schedule else None,
            "num_producers": num_producers if not adjust_schedule else None,
            "drilling_duration": drilling_duration if not adjust_schedule else None,
        })

    if field_type == "Oil Field":
        params_update.update({
            "stoiip_range": stoiip_range,
            "residual_oil_fraction": residual_oil_fraction,
        })
    elif field_type == "Gas Field":
        params_update.update({
            "ogiip_range": ogiip_range,
            "gas_recovery_factor": gas_recovery_factor,
        })
    elif field_type == "Oil Field with WI":
        params_update.update({
            "stoiip_range": stoiip_range,
            "residual_oil_fraction": residual_oil_fraction,
            "maximum_win_rate": max_water_injection_rate,
            "adjust_producer_schedule": adjust_producer_schedule,
            "adjust_injector_schedule": adjust_injector_schedule,
            "manual_producer_schedule": manual_producer_schedule,
            "manual_injector_schedule": manual_injector_schedule,
            "water_injection_rate": water_injection_rate,
            "value_of_water": value_of_water,
            "aquifer_strength_range": aquifer_strength_range,
            "water_injection_well_life": water_injection_well_life,
            "water_injection_efficiency": water_injection_efficiency,
        })
    elif field_type == "Oil Field with WI & GI":
        params_update.update({
            "stoiip_range": stoiip_range,
            "residual_oil_fraction": residual_oil_fraction,
            "maximum_win_rate": max_water_injection_rate,
            "maximum_gin_rate": max_gas_injection_rate,
            "adjust_producer_schedule": adjust_producer_schedule,
            "adjust_water_injector_schedule": adjust_water_injector_schedule,
            "adjust_gas_injector_schedule": adjust_gas_injector_schedule,
            "manual_producer_schedule": manual_producer_schedule,
            "manual_injector_schedule": manual_injector_schedule,
            "manual_gas_injector_schedule": manual_gas_injector_schedule,
            "water_injection_rate": water_injection_rate,
            "value_of_water": value_of_water,
            "aquifer_strength_range": aquifer_strength_range,
            "water_injection_well_life": water_injection_well_life,
            "water_injection_efficiency": water_injection_efficiency,
            "gas_injection_rate": gas_injection_rate,
            "value_of_gas": value_of_gas,
            "gas_injection_well_life": gas_injection_well_life,
            "gas_injection_efficiency": gas_injection_efficiency,
        })

    st.session_state.parameters.update(params_update)

    # # Simulation Placeholder
    # if st.sidebar.button("Run Simulation"):
    #     st.write("Running Simulation for **{}**".format(field_type))

    # ----------------------------
    # Simulation control (Run + Save)
    # ----------------------------
    st.sidebar.subheader("Simulation Control")

    run_clicked = st.sidebar.button(
        "Run Simulation",
        type="primary",
        help="The model will only run when you press this button. If you change any inputs, press it again to re-run.",
    )

    save_clicked = st.sidebar.button(
        "Save Project",
        help="Save all current selected inputs (including parameter ranges and distribution choices) to a JSON project file.",
    )

    if save_clicked:
        payload = {
            "project_name": st.session_state.get("project_name", "Hyperion_Project"),
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "parameters": _json_safe(st.session_state.get("parameters", {})),
        }
        st.session_state["__project_payload__"] = payload
        st.sidebar.success("Project prepared. Choose where to save below.")

    if "__project_payload__" in st.session_state:
        with st.sidebar.expander("Save options", expanded=True):
            save_method = st.radio(
                "Save to:",
                ["Download to my computer", "Save to server path"],
                index=0,
                key="save_method",
            )
            default_name = re.sub(
                r"[^A-Za-z0-9_\-]+",
                "_",
                st.session_state.get("project_name", "Hyperion_Project"),
            ).strip("_") or "Hyperion_Project"
            filename = st.text_input(
                "File name",
                value=f"{default_name}.json",
                key="save_filename",
            )
            data_bytes = json.dumps(
                st.session_state["__project_payload__"],
                indent=2,
                ensure_ascii=False,
            ).encode("utf-8")

            if save_method == "Download to my computer":
                st.download_button(
                    "Download Project JSON",
                    data=data_bytes,
                    file_name=filename,
                    mime="application/json",
                )
            else:
                server_path = st.text_input(
                    "Server folder path (where this app is running)",
                    value=os.getcwd(),
                    key="save_server_path",
                )
                if st.button("Save to server", key="save_to_server_btn"):
                    try:
                        os.makedirs(server_path, exist_ok=True)
                        full_path = os.path.join(server_path, filename)
                        with open(full_path, "wb") as f:
                            f.write(data_bytes)
                        st.success(f"Saved to: {full_path}")
                    except Exception as e:
                        st.error(f"Could not save to server path: {e}")

    # ----------------------------
    # Run gating (independent from Save)
    # ----------------------------
    current_hash = _param_hash(st.session_state.get("parameters", {}))

    if run_clicked:
        st.session_state["__last_run_hash__"] = current_hash
        st.session_state["__has_run__"] = True

    # Require a fresh Run click if inputs changed since last run
    if (not st.session_state.get("__has_run__", False)) or (st.session_state.get("__last_run_hash__") != current_hash):
        st.info("Adjust parameters on the left, then press **Run Simulation**.")
        st.stop()

    # Simulation time
    simulation_years = (project_end_date - project_start_date).days / 365.25
    num_time_steps = max(2, int(np.ceil(simulation_years * 12)) + 1)
    t = np.linspace(0, simulation_years, num_time_steps)
    # st.write(f"Simulation duration: {simulation_years:.2f} years with {len(t)} intervals.")


    #----------------------------------------------------------------------------------------------------------
    # Function to calculate drilling schedule based on manual schedule
    def calculate_drilling_schedule(manual_schedule):
        """
        Create a drilling schedule that distributes wells evenly within each year.
        Handles years with 0 wells by skipping them.
        """
        schedule = []  # List to store start times for all wells
        cumulative_year_offset = 0  # Track the cumulative year offset

        for year, wells in manual_schedule:
            if wells == 0:
                cumulative_year_offset += 1  # Skip this year
                continue  # Go to the next year

            duration_per_well = 1.0 / wells  # Fraction of the year for each well
            start_time_base = cumulative_year_offset  # Start of the current year

            for well in range(wells):
                # Spread wells evenly throughout the year
                start_time = start_time_base + well * duration_per_well
                schedule.append(start_time)

            cumulative_year_offset += 1  # Move to the next year

        return schedule
    #----------------------------------------------------------------------------------------------------------

    # Function for Arps decline
    def arps_decline(t, qi, di, b):
        denominator = 1 + b * di * t
        denominator = np.maximum(denominator, 1e-6)  # Prevent division by zero
        if np.isclose(b, 0):
            return qi * np.exp(-di * t)
        else:
            return qi / denominator ** (1 / b)
    #----------------------------------------------------------------------------------------------------------
        
    # Generate well start times based on field type
    if field_type == "Oil Field" or field_type == "Gas Field":
        # Existing logic for Oil Field and Gas Field
        if adjust_schedule:
            well_start_times = calculate_drilling_schedule(manual_schedule)
        else:
            well_start_times = np.random.uniform(0, simulation_years, size=num_producers[1])
    
    elif field_type == "Oil Field with WI":
        # Separate schedules for producers and water injectors
        producer_start_times = []
        injector_start_times = []
        
        if adjust_producer_schedule:
            # Generate start times for producers
            producer_start_times = calculate_drilling_schedule(manual_producer_schedule)
        else:
            # Randomly distribute producer wells
            producer_start_times = np.random.uniform(0, simulation_years, size=len(manual_producer_schedule))
    
        if adjust_injector_schedule:
            # Generate start times for injectors
            injector_start_times = calculate_drilling_schedule(manual_injector_schedule)
        else:
            # Randomly distribute injector wells
            injector_start_times = np.random.uniform(0, simulation_years, size=len(manual_injector_schedule))
    
    elif field_type == "Oil Field with WI & GI":
        # Separate schedules for producers, water injectors, and gas injectors
        producer_start_times = []
        water_injector_start_times = []
        gas_injector_start_times = []
        
        if adjust_producer_schedule:
            # Generate start times for producers
            producer_start_times = calculate_drilling_schedule(manual_producer_schedule)
        else:
            # Randomly distribute producer wells
            producer_start_times = np.random.uniform(0, simulation_years, size=len(manual_producer_schedule))
    
        if adjust_water_injector_schedule:
            # Generate start times for water injectors
            water_injector_start_times = calculate_drilling_schedule(manual_injector_schedule)
        else:
            # Randomly distribute water injector wells
            water_injector_start_times = np.random.uniform(0, simulation_years, size=len(manual_injector_schedule))
    
        if adjust_gas_injector_schedule:
            # Generate start times for gas injectors
            gas_injector_start_times = calculate_drilling_schedule(manual_gas_injector_schedule)
        else:
            # Randomly distribute gas injector wells
            gas_injector_start_times = np.random.uniform(0, simulation_years, size=len(manual_gas_injector_schedule))
        
        
    #----------------------------------------------------------------------------------------------------------

    #----------------------------------------------------------------------------------------------------------

    def calculate_production_profiles(t, scenario, max_slots, diversify_assumptions, degradation_factor, max_oil_rate):
        num_wells = scenario["num_wells"]
        drilling_duration_range = scenario["drilling_duration"]
        di_range = scenario["decline_rate"]
        b_range = scenario["b"]
        initial_rate_range = scenario["initial_rate"]
        manual_start_times = scenario.get("well_start_times") or []

        well_productions = []  # Store individual well production for each well
        total_production = np.zeros_like(t)  # Initialize total production array
        active_wells = []  # List of currently active wells (indexes)
        well_schedule = []  # Well schedules for Gantt chart
        killed_wells = set()  # Track wells that have been permanently killed

        # Track the current time offset for drilling
        current_time_offset = 0

        # Generate random assumptions if not diversifying
        if not diversify_assumptions:
            rate = sample_range(initial_rate_range, get_dist("dist_initial_rate"))  # Same initial rate for all wells
            di = sample_range(di_range, get_dist("dist_decline")) / 100  # Same decline rate for all wells
            b = sample_range(b_range, get_dist("dist_arpsb"))  # Same Arps b parameter for all wells

        for j in range(num_wells):
            if diversify_assumptions:
                # Generate random assumptions for each well
                rate = sample_range(initial_rate_range, get_dist("dist_initial_rate"))
                di = sample_range(di_range, get_dist("dist_decline")) / 100
                b = sample_range(b_range, get_dist("dist_arpsb"))
            else:
                # Apply degradation factor for wells beyond P1
                if j > 0:
                    rate *= degradation_factor

            # Determine start time for the well
            if adjust_schedule:
                # Use manual schedule well start times
                time_offset = manual_start_times[j] if j < len(manual_start_times) else t[-1]
            else:
                # Default random drilling schedule
                if j == 0:
                    time_offset = 0  # First well starts at the beginning
                else:
                    drilling_duration = np.random.uniform(*drilling_duration_range) / 365.25  # Convert days to years
                    current_time_offset += drilling_duration
                    time_offset = current_time_offset

            # Generate production profile for the well
            production = arps_decline(t - time_offset, rate, di, b)

            # Ensure production starts only after the first well's actual start time
            production[t < time_offset] = 0  # Zero out production before the well starts

            well_productions.append(production)

            # Record well schedule
            well_schedule.append({
                "well": f"P{j+1}",
                "start": time_offset,
                "end": t[-1],  # Default end is the end of simulation unless killed
                "rate": rate,
                "di": di,
                "b": b,
            })

        # Dynamically manage active and inactive wells.
        # A well is only removed for slot capacity when a new scheduled well is ready to enter.
        for i in range(len(t)):
            # Remove wells that have naturally reached their scheduled end.
            active_wells = [w for w in active_wells if well_schedule[w]["end"] >= t[i]]

            ready_wells = [
                j for j in range(num_wells)
                if j not in killed_wells
                and j not in active_wells
                and t[i] >= well_schedule[j]["start"]
            ]

            for j in ready_wells:
                if len(active_wells) >= max_slots:
                    if slot_strategy.startswith("Option A"):
                        kill_pos = 0
                    else:
                        current_rates = [well_productions[w][i] for w in active_wells]
                        kill_pos = int(np.argmin(current_rates))

                    kill_index = active_wells.pop(kill_pos)
                    killed_wells.add(kill_index)
                    well_productions[kill_index][i:] = 0.0
                    well_schedule[kill_index]["end"] = t[i]

                active_wells.append(j)

            production_rate = sum(well_productions[w][i] for w in active_wells)

            if production_rate > max_oil_rate and production_rate > 0:
                scaling_factor = max_oil_rate / production_rate
                for well_index in active_wells:
                    well_productions[well_index][i] *= scaling_factor

            total_production = np.sum(well_productions, axis=0)

        # Ensure killed wells have accurate schedules
        for j in killed_wells:
            if np.any(well_productions[j] > 0):
                last_production_index = np.max(np.where(well_productions[j] > 0))
                well_schedule[j]["end"] = t[last_production_index]
            else:
                well_schedule[j]["end"] = well_schedule[j]["start"]  # No production at all
                
        # Debugging output for scenario if needed (I checked several times)
        # print("\nDebugging Scenario:")
        # for well in well_schedule:
        #     start_date = datetime(2029, 4, 1) + timedelta(days=well["start"] * 365.25)
        #     end_date = datetime(2029, 4, 1) + timedelta(days=well["end"] * 365.25)
        #     life_years = well["end"] - well["start"]
        #     print(f"Well {well['well']}: Starts at {start_date.date()}, ends at {end_date.date()}, life = {life_years:.2f} years.")

        return total_production, len(active_wells), well_schedule, well_productions
#--------------------------------------------------------------------------------------------------------------




# OIL FIELD with WI OPTION
#---------------------------------------------------------------------------------------------------------------
    def calculate_production_profiles_with_WI(t, scenario, max_slots, diversify_assumptions, degradation_factor, max_oil_rate, max_water_injection_rate):
        num_producers = scenario["num_producers"]
        num_water_injectors = scenario["num_injectors"]
        di_range = scenario["decline_rate"]
        b_range = scenario["b"]
        initial_rate_range = scenario["initial_rate"]
        water_injection_rate_range = scenario["water_injection_rate"]
        water_injection_efficiency = scenario["water_injection_efficiency"]
        aquifer_strength_range = scenario["aquifer_strength"]
        water_injection_life_range = scenario["water_injection_well_life"]
        value_of_water_range = scenario["value_of_water"]
    
        producer_productions = []
        water_injections = []
        total_oil_production = np.zeros_like(t)
        total_water_injection = np.zeros_like(t)
        active_producers = []
        active_water_injectors = []
        well_schedule = []
        killed_producers = set()
        killed_water_injectors = set()
        water_injection_rate = sample_range(water_injection_rate_range, get_dist("dist_wi_rate"))
        aquifer_support = sample_range(aquifer_strength_range, get_dist("dist_aquifer"))
        water_injection_life = np.random.uniform(*water_injection_life_range)
        value_of_water = sample_range(value_of_water_range, get_dist("dist_vow"))
    
        # Generate random assumptions if not diversifying
        if not diversify_assumptions:
            rate = sample_range(initial_rate_range, get_dist("dist_initial_rate"))
            di = sample_range(di_range, get_dist("dist_decline")) / 100
            b = sample_range(b_range, get_dist("dist_arpsb"))

    
        producer_start_times = scenario.get("producer_start_times", [0] * num_producers)
        injector_start_times = scenario.get("injector_start_times", [0] * num_water_injectors)
    
        # Process Producers
        for j in range(num_producers):
            if diversify_assumptions:
                rate = sample_range(initial_rate_range, get_dist("dist_initial_rate"))
                di = sample_range(di_range, get_dist("dist_decline")) / 100
                b = sample_range(b_range, get_dist("dist_arpsb"))
            else:
                if j > 0:
                    rate *= degradation_factor
    
            time_offset = producer_start_times[j]
            production = arps_decline(t - time_offset, rate, di, b)
            production[t < time_offset] = 0
            producer_productions.append(production)
    
            well_schedule.append({
                "well": f"P{j+1}",
                "start": time_offset,
                "end": t[-1],
                "rate": rate,
                "di": di,
                "b": b,
            })
    
        # Process Water Injectors
        for j in range(num_water_injectors):
            if diversify_assumptions:
                water_injection_rate = sample_range(water_injection_rate_range, get_dist("dist_wi_rate"))
                water_injection_life = np.random.uniform(*water_injection_life_range)
            else:
                if j > 0:
                    water_injection_rate *= degradation_factor
                water_injection_life = np.random.uniform(*water_injection_life_range)
    
            time_offset = injector_start_times[j]
            end_time = time_offset + water_injection_life
            injection = np.full_like(t, water_injection_rate)
            injection[t < time_offset] = 0
            injection[t > end_time] = 0
            water_injections.append(injection)

    
            well_schedule.append({
                "well": f"WI{j+1}",
                "start": time_offset,
                "end": end_time,
                "rate": water_injection_rate,
            })
            
        # Combine killed wells
        killed_wells = killed_producers.union(killed_water_injectors)
        active_wells = active_producers + active_water_injectors
    
        # Slot constraints and production updates.
        for i in range(len(t)):
            # Remove wells whose scheduled life has ended.
            active_wells = [w for w in active_wells if well_schedule[w]["end"] >= t[i]]

            ready_wells = [
                j for j in range(num_producers + num_water_injectors)
                if j not in killed_wells
                and j not in active_wells
                and t[i] >= well_schedule[j]["start"]
            ]

            for j in ready_wells:
                if len(active_wells) >= max_slots:
                    if slot_strategy.startswith("Option A"):
                        kill_pos = 0
                    else:
                        current_vals = []
                        for w in active_wells:
                            if w < num_producers:
                                current_vals.append(producer_productions[w][i])
                            else:
                                current_vals.append(water_injections[w - num_producers][i])
                        kill_pos = int(np.argmin(current_vals))

                    kill_index = active_wells.pop(kill_pos)
                    killed_wells.add(kill_index)
                    if kill_index < num_producers:
                        producer_productions[kill_index][i:] = 0.0
                    else:
                        water_injections[kill_index - num_producers][i:] = 0.0
                    well_schedule[kill_index]["end"] = t[i]

                active_wells.append(j)

            current_active_producers = [w for w in active_wells if w < num_producers]
            current_active_water_injectors = [w for w in active_wells if w >= num_producers]

            total_production = sum(producer_productions[w][i] for w in current_active_producers)
            total_injection = sum(
                water_injections[w - num_producers][i]
                for w in current_active_water_injectors
            )

            # Apply the injection facility limit before calculating the oil benefit.
            if total_injection > max_water_injection_rate and total_injection > 0:
                scaling_factor = max_water_injection_rate / total_injection
                for well_index in current_active_water_injectors:
                    water_injections[well_index - num_producers][i] *= scaling_factor
                total_injection = max_water_injection_rate

            water_injection_impact = 0.0
            if current_active_producers and total_production > 0:
                water_injection_impact = (
                    aquifer_support
                    * total_injection
                    * water_injection_efficiency
                    * value_of_water
                )

                # Allocate injection uplift proportionally to active producer rates so
                # well-by-well profiles reconcile with the total field profile.
                base_rates = np.array(
                    [producer_productions[w][i] for w in current_active_producers],
                    dtype=float,
                )
                base_sum = float(np.sum(base_rates))
                if base_sum > 0 and water_injection_impact > 0:
                    weights = base_rates / base_sum
                    for well_index, weight in zip(current_active_producers, weights):
                        producer_productions[well_index][i] += water_injection_impact * weight

            total_production = sum(
                producer_productions[w][i] for w in current_active_producers
            )

            if total_production > max_oil_rate and total_production > 0:
                scaling_factor = max_oil_rate / total_production
                for well_index in current_active_producers:
                    producer_productions[well_index][i] *= scaling_factor
                total_production = max_oil_rate

            total_oil_production[i] = total_production
            total_water_injection[i] = total_injection

        active_producers = [w for w in active_wells if w < num_producers]
        active_water_injectors = [w for w in active_wells if w >= num_producers]

        return (
            total_oil_production,
            total_water_injection,
            len(active_producers),
            len(active_water_injectors),
            well_schedule,
            producer_productions,
            water_injections,
        )

#---------------------------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------------------------------
# OIL FIELD with WI and GI OPTION
#---------------------------------------------------------------------------------------------------------------

    def calculate_production_profiles_with_WI_GI(
            t, scenario, max_slots, diversify_assumptions, degradation_factor, max_oil_rate, max_water_injection_rate, max_gas_injection_rate):
        
        num_producers = scenario["num_producers"]
        num_water_injectors = scenario["num_water_injectors"]
        num_gas_injectors = scenario["num_gas_injectors"]
        di_range = scenario["decline_rate"]
        b_range = scenario["b"]
        initial_rate_range = scenario["initial_rate"]
        water_injection_rate_range = scenario["water_injection_rate"]
        gas_injection_rate_range = scenario["gas_injection_rate"]
        water_injection_efficiency = scenario["water_injection_efficiency"]
        gas_injection_efficiency = scenario["gas_injection_efficiency"]
        aquifer_strength_range = scenario["aquifer_strength"]
        water_injection_life_range = scenario["water_injection_life"]
        gas_injection_life_range = scenario["gas_injection_life"]
        value_of_water_range = scenario["value_of_water"]  # Add value of water to the scenario
        value_of_gas_range = scenario["value_of_gas"]  # Add value of water to the scenario
    
        producer_productions = []  # Store individual producer production for each well
        water_injections = []  # Store individual water injection rates for each injector
        gas_injections = []  # Store individual gas injection rates for each injector
        total_oil_production = np.zeros_like(t)  # Initialize total oil production array
        total_water_injection = np.zeros_like(t)  # Initialize total water injection array
        total_gas_injection = np.zeros_like(t)  # Initialize total gas injection array
        active_producers = []  # List of currently active producers
        active_water_injectors = []  # List of currently active water injectors
        active_gas_injectors = []  # List of currently active gas injectors
        well_schedule = []  # Well schedules for Gantt chart
        killed_producers = set()  # Track producers that have been permanently killed
        killed_water_injectors = set()  # Track water injectors that have been permanently killed
        killed_gas_injectors = set()  # Track gas injectors that have been permanently killed
        
        water_injection_rate = sample_range(water_injection_rate_range, get_dist("dist_wi_rate"))  # Same injection rate
        gas_injection_rate = sample_range(gas_injection_rate_range, get_dist("dist_gi_rate"))  # Same gas injection rate
        aquifer_support = sample_range(aquifer_strength_range, get_dist("dist_aquifer"))  # Same aquifer strength
        water_injection_life = np.random.uniform(*water_injection_life_range)  # Same water injection life
        gas_injection_life = np.random.uniform(*gas_injection_life_range)  # Same gas injection life
        value_of_water= sample_range(value_of_water_range, get_dist("dist_vow"))
        value_of_gas= sample_range(value_of_gas_range, get_dist("dist_vog"))
    
        # Generate random assumptions if not diversifying
        if not diversify_assumptions:
            rate = sample_range(initial_rate_range, get_dist("dist_initial_rate"))  # Same initial rate for all wells
            di = sample_range(di_range, get_dist("dist_decline")) / 100  # Same decline rate for all wells
            b = sample_range(b_range, get_dist("dist_arpsb"))  # Same Arps b parameter for all wells
          
        producer_start_times_local = scenario.get("producer_start_times", [0] * num_producers)
        water_injector_start_times_local = scenario.get("water_injector_start_times", [0] * num_water_injectors)
        gas_injector_start_times_local = scenario.get("gas_injector_start_times", [0] * num_gas_injectors)

        # Calculate producer schedules
        for j in range(num_producers):
            if diversify_assumptions:
                rate = sample_range(initial_rate_range, get_dist("dist_initial_rate"))
                di = sample_range(di_range, get_dist("dist_decline")) / 100
                b = sample_range(b_range, get_dist("dist_arpsb"))
            else:
                if j > 0:
                    rate *= degradation_factor
    
            time_offset = producer_start_times_local[j] if j < len(producer_start_times_local) else t[-1]
            production = arps_decline(t - time_offset, rate, di, b)
            production[t < time_offset] = 0
            producer_productions.append(production)
    
            well_schedule.append({
                "well": f"P{j+1}",
                "start": time_offset,
                "end": t[-1],
                "rate": rate,
                "di": di,
                "b": b,
            })
    
        # Calculate water injector schedules
        for j in range(num_water_injectors):
            if diversify_assumptions:
                water_injection_rate = sample_range(water_injection_rate_range, get_dist("dist_wi_rate"))
                water_injection_life = np.random.uniform(*water_injection_life_range)
            else:
                if j > 0:
                    water_injection_rate *= degradation_factor
                water_injection_life = np.random.uniform(*water_injection_life_range)
    
            time_offset = water_injector_start_times_local[j] if j < len(water_injector_start_times_local) else t[-1]
            # end_time = time_offset + water_injection_life * 365.25
            end_time = time_offset + water_injection_life
            injection = np.full_like(t, water_injection_rate)
            injection[t < time_offset] = 0
            injection[t > end_time] = 0
            water_injections.append(injection)
    
            well_schedule.append({
                "well": f"WI{j+1}",
                "start": time_offset,
                "end": end_time,
                "rate": water_injection_rate,
            })
    
        # Calculate gas injector schedules
        for j in range(num_gas_injectors):
            if diversify_assumptions:
                gas_injection_rate = sample_range(gas_injection_rate_range, get_dist("dist_gi_rate"))
                gas_injection_life = np.random.uniform(*gas_injection_life_range)
            else:
                if j > 0:
                    gas_injection_rate *= degradation_factor
                gas_injection_life = np.random.uniform(*gas_injection_life_range)
    
            time_offset = gas_injector_start_times_local[j] if j < len(gas_injector_start_times_local) else t[-1]
            # end_time = time_offset + gas_injection_life * 365.25
            end_time = time_offset + gas_injection_life  # Lifetime in days, constrained to the end of simulation
            injection = np.full_like(t, gas_injection_rate)
            injection[t < time_offset] = 0
            injection[t > end_time] = 0
            gas_injections.append(injection)
    
            well_schedule.append({
                "well": f"GI{j+1}",
                "start": time_offset,
                "end": end_time,
                "rate": gas_injection_rate,
            })
    
        # Combine killed producers, water injectors, and gas injectors
        killed_wells = killed_producers.union(killed_water_injectors).union(killed_gas_injectors)
        active_wells = active_producers + active_water_injectors + active_gas_injectors
    
        # Apply slot constraints for each time step
        # last_active_well_end = 0
        # for i in range(len(t)):
        #     # Check and update active producers, water injectors, and gas injectors
        #     if len(active_wells) >= max_slots:
        #         oldest_well_index = active_wells.pop(0)
        #         killed_wells.add(oldest_well_index)
    
        #         if oldest_well_index < num_producers:
        #             producer_productions[oldest_well_index][i:] = 0
        #         elif oldest_well_index < num_producers + num_water_injectors:
        #             water_injections[oldest_well_index - num_producers][i:] = 0
        #         else:
        #             gas_injections[oldest_well_index - num_producers - num_water_injectors][i:] = 0
    
        #         well_schedule[oldest_well_index]["end"] = t[i]
        #         last_active_well_end = t[i]
                
        for i in range(len(t)):
            # Remove wells whose scheduled life has ended.
            active_wells = [w for w in active_wells if well_schedule[w]["end"] >= t[i]]

            ready_wells = [
                j for j in range(num_producers + num_water_injectors + num_gas_injectors)
                if j not in killed_wells
                and j not in active_wells
                and t[i] >= well_schedule[j]["start"]
            ]

            for j in ready_wells:
                if len(active_wells) >= max_slots:
                    if slot_strategy.startswith("Option A"):
                        kill_pos = 0
                    else:
                        current_vals = []
                        for w in active_wells:
                            if w < num_producers:
                                current_vals.append(producer_productions[w][i])
                            elif w < num_producers + num_water_injectors:
                                current_vals.append(water_injections[w - num_producers][i])
                            else:
                                current_vals.append(
                                    gas_injections[w - num_producers - num_water_injectors][i]
                                )
                        kill_pos = int(np.argmin(current_vals))

                    kill_index = active_wells.pop(kill_pos)
                    killed_wells.add(kill_index)

                    if kill_index < num_producers:
                        producer_productions[kill_index][i:] = 0.0
                    elif kill_index < num_producers + num_water_injectors:
                        water_injections[kill_index - num_producers][i:] = 0.0
                    else:
                        gas_injections[
                            kill_index - num_producers - num_water_injectors
                        ][i:] = 0.0

                    well_schedule[kill_index]["end"] = t[i]

                active_wells.append(j)

            current_active_producers = [
                w for w in active_wells if w < num_producers
            ]
            current_active_water_injectors = [
                w for w in active_wells
                if num_producers <= w < num_producers + num_water_injectors
            ]
            current_active_gas_injectors = [
                w for w in active_wells
                if w >= num_producers + num_water_injectors
            ]

            total_production = sum(
                producer_productions[w][i] for w in current_active_producers
            )
            current_water_injection = sum(
                water_injections[w - num_producers][i]
                for w in current_active_water_injectors
            )
            current_gas_injection = sum(
                gas_injections[w - num_producers - num_water_injectors][i]
                for w in current_active_gas_injectors
            )

            # Apply facility limits separately; water and gas have different units.
            if current_water_injection > max_water_injection_rate and current_water_injection > 0:
                scaling_factor = max_water_injection_rate / current_water_injection
                for well_index in current_active_water_injectors:
                    water_injections[well_index - num_producers][i] *= scaling_factor
                current_water_injection = max_water_injection_rate

            if current_gas_injection > max_gas_injection_rate and current_gas_injection > 0:
                scaling_factor = max_gas_injection_rate / current_gas_injection
                for well_index in current_active_gas_injectors:
                    gas_injections[
                        well_index - num_producers - num_water_injectors
                    ][i] *= scaling_factor
                current_gas_injection = max_gas_injection_rate

            total_injection_uplift = 0.0
            if current_active_producers and total_production > 0:
                water_injection_impact = (
                    aquifer_support
                    * current_water_injection
                    * water_injection_efficiency
                    * value_of_water
                )
                gas_injection_impact = (
                    current_gas_injection
                    * gas_injection_efficiency
                    * value_of_gas
                )
                total_injection_uplift = water_injection_impact + gas_injection_impact

                base_rates = np.array(
                    [producer_productions[w][i] for w in current_active_producers],
                    dtype=float,
                )
                base_sum = float(np.sum(base_rates))
                if base_sum > 0 and total_injection_uplift > 0:
                    weights = base_rates / base_sum
                    for well_index, weight in zip(current_active_producers, weights):
                        producer_productions[well_index][i] += total_injection_uplift * weight

            total_production = sum(
                producer_productions[w][i] for w in current_active_producers
            )

            if total_production > max_oil_rate and total_production > 0:
                scaling_factor = max_oil_rate / total_production
                for well_index in current_active_producers:
                    producer_productions[well_index][i] *= scaling_factor
                total_production = max_oil_rate

            total_oil_production[i] = total_production
            total_water_injection[i] = current_water_injection
            total_gas_injection[i] = current_gas_injection

        active_producers = [w for w in active_wells if w < num_producers]
        active_water_injectors = [
            w for w in active_wells
            if num_producers <= w < num_producers + num_water_injectors
        ]
        active_gas_injectors = [
            w for w in active_wells
            if w >= num_producers + num_water_injectors
        ]

        return (total_oil_production, total_water_injection, total_gas_injection, len(active_producers), len(active_water_injectors), len(active_gas_injectors), well_schedule,
                producer_productions, water_injections, gas_injections)


#---------------------------------------------------------------------------------------------------------------
    # Initialize scenarios
    scenarios = []
    for _ in range(simulation_number): 
        if field_type == "Oil Field" or field_type == "Gas Field":
            if adjust_schedule:
                well_start_times = calculate_drilling_schedule(manual_schedule)
                num_wells = len(well_start_times)  # Use the number of wells from manual schedule
            else:
                num_wells = np.random.randint(num_producers[0], num_producers[1] + 1)  # inclusive slider range
                well_start_times = np.random.uniform(0, simulation_years, size=num_producers[1])
    
            scenario = {
                "initial_rate": initial_rate,
                "decline_rate": yearly_decline,
                "b": arps_b,
                "num_wells": num_wells,
                "drilling_duration": drilling_duration,
                "incremental_factor": sample_range(incremental_factor, get_dist("dist_incremental")),
                "well_start_times": well_start_times if adjust_schedule else None,
            }
    
        elif field_type == "Oil Field with WI":
            if adjust_producer_schedule:
                producer_start_times = calculate_drilling_schedule(manual_producer_schedule)
                num_producers = len(producer_start_times)  # Number of producer wells
            else:
                num_producers = np.random.randint(*num_producers)  # Use range from the slider
                producer_start_times = np.random.uniform(0, simulation_years, size=num_producers)
    
            if adjust_injector_schedule:
                injector_start_times = calculate_drilling_schedule(manual_injector_schedule)
                num_injectors = len(injector_start_times)  # Number of injector wells
            else:
                num_injectors = np.random.randint(*num_injectors)  # Use range from the slider
                injector_start_times = np.random.uniform(0, simulation_years, size=num_injectors)
    
            scenario = {
                "initial_rate": initial_rate,
                "decline_rate": yearly_decline,
                "b": arps_b,
                "num_producers": num_producers,
                "num_injectors": num_injectors,
                "drilling_duration": drilling_duration,
                "water_injection_rate": water_injection_rate,
                "water_injection_efficiency": water_injection_efficiency,
                "aquifer_strength": aquifer_strength_range,  # Add aquifer strength parameter here
                "incremental_factor": sample_range(incremental_factor, get_dist("dist_incremental")),
                "water_injection_well_life": water_injection_well_life,  # Add the slider value here
                "value_of_water": value_of_water,  # Add the slider value here
                "producer_start_times": producer_start_times,  # Ensure this is included
                "injector_start_times": injector_start_times,  # Ensure this is included
            }
    
        elif field_type == "Oil Field with WI & GI":
            if adjust_producer_schedule:
                producer_start_times = calculate_drilling_schedule(manual_producer_schedule)
                num_producers = len(producer_start_times)  # Number of producer wells
            else:
                num_producers = np.random.randint(*num_producers)  # Use range from the slider
                producer_start_times = np.random.uniform(0, simulation_years, size=num_producers)
    
            if adjust_water_injector_schedule:
                water_injector_start_times = calculate_drilling_schedule(manual_injector_schedule)
                num_water_injectors = len(water_injector_start_times)  # Number of water injectors
            else:
                num_water_injectors = np.random.randint(*num_injectors)  # Use range from the slider
                water_injector_start_times = np.random.uniform(0, simulation_years, size=num_water_injectors)
    
            if adjust_gas_injector_schedule:
                gas_injector_start_times = calculate_drilling_schedule(manual_gas_injector_schedule)
                num_gas_injectors = len(gas_injector_start_times)  # Number of gas injectors
            else:
                num_gas_injectors = np.random.randint(*num_gas_injectors)  # Use range from the slider
                gas_injector_start_times = np.random.uniform(0, simulation_years, size=num_gas_injectors)
    
            scenario = {
                "initial_rate": initial_rate,
                "decline_rate": yearly_decline,
                "b": arps_b,
                "num_producers": num_producers,
                "num_water_injectors": num_water_injectors,
                "num_gas_injectors": num_gas_injectors,
                "drilling_duration": drilling_duration,
                "water_injection_rate": water_injection_rate,
                "water_injection_efficiency": water_injection_efficiency,
                "gas_injection_rate": gas_injection_rate,
                "gas_injection_efficiency": gas_injection_efficiency,
                "incremental_factor": sample_range(incremental_factor, get_dist("dist_incremental")),
                "aquifer_strength": aquifer_strength_range,  # Add aquifer strength parameter here
                'water_injection_life':water_injection_well_life,
                'gas_injection_life':gas_injection_well_life,
                "value_of_water": value_of_water,
                "value_of_gas": value_of_gas,
                "producer_start_times": producer_start_times,
                "water_injector_start_times": water_injector_start_times,
                "gas_injector_start_times": gas_injector_start_times,
            }
    
        scenarios.append(scenario)

    
    # Initialize progress bar and percentage display
    progress_bar = st.progress(0)
    progress_text = st.empty()
    total_steps = len(scenarios)
    current_progress = 0
    
    # Simulation Loop
    production_profiles = []
    cumulative_productions = []
    incremental_productions = []
    active_well_counts = []
    well_schedules = []
    all_well_productions = []
    all_water_injection_profiles = []
    all_gas_injection_profiles = []
    
    for idx, scenario in enumerate(scenarios):
        # Calculate production profiles based on the field type
        if field_type == "Oil Field with WI":
            total_production, total_water_injection, active_producers, active_injectors, well_schedule, well_productions, water_injections = calculate_production_profiles_with_WI(
                t, scenario, max_slots, diversify_assumptions, degradation_factor, maximum_rate, max_water_injection_rate
            )
            all_water_injection_profiles.append(water_injections)

    
        elif field_type == "Oil Field with WI & GI":
            total_production, total_water_injection, total_gas_injection, active_producers, active_water_injectors, active_gas_injectors, well_schedule, well_productions, water_injections, gas_injections = calculate_production_profiles_with_WI_GI(
                t, scenario, max_slots, diversify_assumptions, degradation_factor, maximum_rate, max_water_injection_rate, max_gas_injection_rate
            )
            all_water_injection_profiles.append(water_injections)
            all_gas_injection_profiles.append(gas_injections)
    
        else:
            total_production, active_wells, well_schedule, well_productions = calculate_production_profiles(
                t, scenario, max_slots, diversify_assumptions, degradation_factor, maximum_rate
            )
    
        # Append simulation results
        production_profiles.append(total_production)
        cumulative_productions.append(np.cumsum(total_production * 30.4375) / 1000)  # Convert to MMSTB
        incremental_productions.append(cumulative_productions[-1] * scenario.get("incremental_factor", 1.0))
    
        # Count active wells
        active_well_count_time_series = [
            len([well for well in well_schedule if well["start"] <= time and well["end"] >= time])
            for time in t
        ]
        active_well_count_time_series = np.clip(active_well_count_time_series, 0, max_slots)  # Enforce slot constraints
        active_well_counts.append(active_well_count_time_series)
    
        # Append well schedules and production details
        well_schedules.append(well_schedule)
        all_well_productions.append(well_productions)
    
        # Update progress bar and text
        current_progress += 1
        percentage = int(current_progress / total_steps * 100)
        progress_bar.progress(percentage)
        progress_text.text(f"Progress: {percentage}%")
    
    # SIMULATION COMPLETED

        
#---------------------------------------------------------------------------------------------------------------
  
   # Convert `active_well_counts` to a 2D array
    # Ensure active_well_counts is not empty and convert to a NumPy array
    if not active_well_counts:
        st.error("Error: active_well_counts is empty. Ensure that the simulation data has been correctly generated.")
    else:
        active_well_counts = np.array(active_well_counts)
    
    # Additional logic for Oil Field with WI or WI and GI to handle separate producer and injector counts
    if field_type in ["Oil Field with WI", "Oil Field with WI & GI"]:
        if not well_schedules:
            st.error("Error: well_schedules is empty. Ensure that the simulation data has been correctly generated.")
        else:
            producer_well_counts = np.array([
                [len([well for well in well_schedule if well["start"] <= time and well["end"] >= time and str(well["well"]).startswith("P")]) for time in t]
                for well_schedule in well_schedules
            ])
    
            water_injector_counts = np.array([
                [len([well for well in well_schedule if well["start"] <= time and well["end"] >= time and str(well["well"]).startswith("WI")]) for time in t]
                for well_schedule in well_schedules
            ])
    
            # Handle gas injectors if "Oil Field with WI & GI" option is selected
            if field_type == "Oil Field with WI & GI":
                gas_injector_counts = np.array([
                    [len([well for well in well_schedule if well["start"] <= time and well["end"] >= time and str(well["well"]).startswith("GI")]) for time in t]
                    for well_schedule in well_schedules
                ])
    
    # Validate the shape of active_well_counts
    if isinstance(active_well_counts, np.ndarray) and active_well_counts.ndim == 2:
        if active_well_counts.shape[1] != len(t):
            st.error(f"Shape mismatch: active_well_counts must have shape (n_scenarios, len(t)), but got {active_well_counts.shape}.")
    else:
        st.error("Error: active_well_counts is not a valid 2D array.")
    
    # Validate producer and injector counts for "Oil Field with WI" and "Oil Field with WI & GI"
    if field_type in ["Oil Field with WI", "Oil Field with WI & GI"]:
        if isinstance(producer_well_counts, np.ndarray) and producer_well_counts.ndim == 2:
            if producer_well_counts.shape[1] != len(t):
                st.error(f"Shape mismatch: producer_well_counts must have shape (n_scenarios, len(t)), but got {producer_well_counts.shape}.")
        else:
            st.error("Error: producer_well_counts is not a valid 2D array.")
    
        if isinstance(water_injector_counts, np.ndarray) and water_injector_counts.ndim == 2:
            if water_injector_counts.shape[1] != len(t):
                st.error(f"Shape mismatch: water_injector_counts must have shape (n_scenarios, len(t)), but got {water_injector_counts.shape}.")
        else:
            st.error("Error: water_injector_counts is not a valid 2D array.")
    
        if field_type == "Oil Field with WI & GI":
            if isinstance(gas_injector_counts, np.ndarray) and gas_injector_counts.ndim == 2:
                if gas_injector_counts.shape[1] != len(t):
                    st.error(f"Shape mismatch: gas_injector_counts must have shape (n_scenarios, len(t)), but got {gas_injector_counts.shape}.")
            else:
                st.error("Error: gas_injector_counts is not a valid 2D array.")

        


    # Compute P90, P50, P10
    def compute_percentiles(data):
        # Ensure the input data is a NumPy array
        if isinstance(data, list):
            data = np.array(data)
        # Check if the array is empty
        if data.size == 0:
            return np.zeros((3, len(t)))
        return np.percentile(data, [10, 50, 90], axis=0)



    P90_prod, P50_prod, P10_prod = compute_percentiles(production_profiles)
    P90_cum, P50_cum, P10_cum = compute_percentiles(cumulative_productions)
    P90_incr, P50_incr, P10_incr = compute_percentiles(incremental_productions)
    P90_wells, P50_wells, P10_wells = compute_percentiles(active_well_counts)
    if field_type == "Oil Field with WI":
        # Calculate Total Water Injection Profiles
        total_water_injection_profiles = np.array([np.sum(water_injections, axis=0) for water_injections in all_water_injection_profiles])

        # Compute P90, P50, P10 for Total Water Injection
        P90_water_injection, P50_water_injection, P10_water_injection = compute_percentiles(total_water_injection_profiles)
    
        # Function to plot water injection rates
        def plot_water_injection_rates(t, profiles, P90, P50, P10):
            """
            Plot water injection rate profiles with P90, P50, P10 and scenario cloud.
            """
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot each scenario as a cloud
            for profile in profiles:
                ax.plot(t, profile, color="deepskyblue", alpha=0.1)
            
            # Plot P90, P50, P10 lines
            ax.plot(t, P90, color="red", linewidth=2, label="P90")
            ax.plot(t, P50, color="black", linewidth=2, label="P50")
            ax.plot(t, P10, color="purple", linewidth=2, label="P10")
            
            # Title, labels, and legend
            ax.set_title("Total Water Injection Rate (Mstb/d)")
            ax.set_xlabel("Time (years)")
            ax.set_ylabel("Water Injection Rate (Mstb/d)")
            ax.legend()
            ax.grid()
            st.pyplot(fig)
            plt.close(fig)
        
        # Plot Total Water Injection Rate (Mstb/d)
        plot_water_injection_rates(
            t,
            total_water_injection_profiles,
            P90_water_injection,
            P50_water_injection,
            P10_water_injection
        )



    # # Function to plot P50 scenario well-by-well water injection rates
    # def plot_p50_well_by_well_injection(t, p50_scenario_water_injections):
    #     """
    #     Plot well-by-well water injection rates for the P50 scenario.
    #     """
    #     fig, ax = plt.subplots(figsize=(12, 8))
    
    #     # Plot each well's water injection profile
    #     for well_idx, well_injection_profile in enumerate(p50_scenario_water_injections):
    #         ax.plot(t, well_injection_profile, label=f"WI{well_idx + 1}")
    
    #     # Title, labels, and legend
    #     ax.set_title("P50 Scenario Well-by-Well Water Injection Rate")
    #     ax.set_xlabel("Time (years)")
    #     ax.set_ylabel("Water Injection Rate (Mstb/d)")
    #     ax.legend(loc="upper right", fontsize="small")
    #     ax.grid()
    
    #     # Display the plot
    #     st.pyplot(fig)
    
    # # Extract P50 scenario water injection profiles
    # # Assuming `all_water_injection_profiles` contains individual well injection profiles for all scenarios
    # p50_index = np.argsort([np.sum(np.sum(water_injections, axis=0)) for water_injections in all_water_injection_profiles])[len(all_water_injection_profiles) // 2]
    # p50_scenario_water_injections = all_water_injection_profiles[p50_index]  # Select P50 scenario
    
    # # Plot the well-by-well injection profiles for the P50 scenario
    # plot_p50_well_by_well_injection(t, p50_scenario_water_injections)





    if field_type == "Oil Field with WI & GI":
        # Calculate Total Water Injection Profiles
        total_water_injection_profiles = np.array([
            np.sum(water_injections, axis=0) for water_injections in all_water_injection_profiles
        ])
        
        # Compute P90, P50, P10 for Total Water Injection
        P90_water_injection, P50_water_injection, P10_water_injection = compute_percentiles(total_water_injection_profiles)
    
        # Calculate Total Gas Injection Profiles
        total_gas_injection_profiles = np.array([
            np.sum(gas_injections, axis=0) for gas_injections in all_gas_injection_profiles
        ])
        
        # Compute P90, P50, P10 for Total Gas Injection
        P90_gas_injection, P50_gas_injection, P10_gas_injection = compute_percentiles(total_gas_injection_profiles)
    
        # Function to plot water injection rates
        def plot_water_injection_rates(t, profiles, P90, P50, P10):
            st.header("Total Water Injection Profile")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot each scenario as a cloud
            for profile in profiles:
                ax.plot(t, profile, color="deepskyblue", alpha=0.1)
            
            # Plot P90, P50, P10 lines
            ax.plot(t, P90, color="red", linewidth=2, label="P90")
            ax.plot(t, P50, color="black", linewidth=2, label="P50")
            ax.plot(t, P10, color="purple", linewidth=2, label="P10")
            
            # Title, labels, and legend
            ax.set_title("Total Water Injection Rate (Mstb/d)")
            ax.set_xlabel("Time (years)")
            ax.set_ylabel("Water Injection Rate (Mstb/d)")
            ax.legend()
            ax.grid()
            st.pyplot(fig)
            plt.close(fig)
    
        # Function to plot gas injection rates
        def plot_gas_injection_rates(t, profiles, P90, P50, P10):
            st.header("Total Gas Injection Profile")
            fig, ax = plt.subplots(figsize=(10, 6))
            for profile in profiles:
                ax.plot(t, profile, color="red", alpha=0.1)
            
            # Plot P90, P50, P10 lines
            ax.plot(t, P90, color="red", linewidth=2, label="P90")
            ax.plot(t, P50, color="black", linewidth=2, label="P50")
            ax.plot(t, P10, color="purple", linewidth=2, label="P10")
            
            # Title, labels, and legend
            ax.set_title("Total Gas Injection Rate (MMscf/d)")
            ax.set_xlabel("Time (years)")
            ax.set_ylabel("Gas Injection Rate (MMscf/d)")
            ax.legend()
            ax.grid()
            st.pyplot(fig)
            plt.close(fig)
        
        # Plot Total Water Injection Rate (Mstb/d)
        plot_water_injection_rates(
            t,
            total_water_injection_profiles,
            P90_water_injection,
            P50_water_injection,
            P10_water_injection
        )
        
        # Plot Total Gas Injection Rate (MMscf/d)
        plot_gas_injection_rates(
            t,
            total_gas_injection_profiles,
            P90_gas_injection,
            P50_gas_injection,
            P10_gas_injection
        )





    # Function to Describe Scenario (initially tested works fine)
    def describe_scenario(scenario, well_schedule, well_productions):
        description = f"Number of Producers: {len(well_schedule)}\n"
        description += "Well Details:\n"
        for i, well in enumerate(well_schedule):
            start_date = project_start_date + timedelta(days=well["start"] * 365.25)
            drilling_duration_days = int((well["end"] - well["start"]) * 365.25)
            cumulative_oil = np.sum(well_productions[i] * 30.4375) / 1000  # Convert to MMSTB
            description += (
                f"  P{i+1}: Oil Rate={well['rate']:.2f}, "
                f"Decline={well['di'] * 100:.2f}%, "
                f"b={well['b']:.2f}, "
                f"Start Date={start_date}, "
                f"Drilling Duration={drilling_duration_days} days, "
                f"Cumulative Oil={cumulative_oil:.2f} MMSTB\n"
            )
        description += f"Incremental Factor: {scenario['incremental_factor']:.2f}\n"
        return description




    #Calculate peak oil rates and put on profile as lable
    def calculate_peak_rates_and_times(production_profiles, t):
        peak_rates = np.max(production_profiles, axis=1)  # Peak oil rates for each scenario
        peak_times = [t[np.argmax(profile)] for profile in production_profiles]  # Times of peak oil rates
        return peak_rates, peak_times

    # Calculate peak oil rates and times for P90, P50, P10
    peak_rates, peak_times = calculate_peak_rates_and_times(production_profiles, t)
    P90_peak_rate = np.percentile(peak_rates, 10)
    P50_peak_rate = np.percentile(peak_rates, 50)
    P10_peak_rate = np.percentile(peak_rates, 90)
    P90_peak_time = t[np.argmax(P90_prod)]
    P50_peak_time = t[np.argmax(P50_prod)]
    P10_peak_time = t[np.argmax(P10_prod)]


    # Plot Total Production Profile with Peak Oil rate callout
    fig, ax = plt.subplots(figsize=(10, 6))
    for profile in production_profiles:
        # ax.plot(t, profile, color="lightblue", alpha=0.1)
        ax.plot(t, profile, color=profile_color, alpha=0.1)  # Use dynamic color    
    # P90, P50, P10 lines
    ax.plot(t, P90_prod, color="red", linewidth=2, label=f"P90 {rate_label}")
    ax.plot(t, P50_prod, color="black", linewidth=2, label=f"P50 {rate_label}")
    ax.plot(t, P10_prod, color="purple", linewidth=2, label=f"P10 {rate_label}")

    # Add callouts for peak rates at the exact peak positions
    ax.text(P90_peak_time, P90_peak_rate, f"{P90_peak_rate:.2f}", color="red", fontsize=10,
            bbox=dict(facecolor="white", edgecolor="red", boxstyle="round,pad=0.5"))
    ax.text(P50_peak_time, P50_peak_rate, f"{P50_peak_rate:.2f}", color="black", fontsize=10,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"))
    ax.text(P10_peak_time, P10_peak_rate, f"{P10_peak_rate:.2f}", color="purple", fontsize=10,
            bbox=dict(facecolor="white", edgecolor="purple", boxstyle="round,pad=0.5"))

    # Final plot adjustments
    st.header("Total Production Profile")
    # ax.set_title("Total Production Profile (P90, P50, P10)")
    # ax.set_xlabel("Time (years)")
    # ax.set_ylabel("Oil Rate (Mstb/d)")
    # ax.legend()
    # ax.grid()
    
    # Use dynamic labels
    ax.set_title(f"Total {title_suffix} Production Profile")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel(rate_label)
    ax.legend()
    ax.grid()
    st.pyplot(fig)
    plt.close(fig)


    # Cumulative production with callouts
    def plot_with_callouts(t, profiles, P90, p50, P10, title, xlabel, ylabel, colors, labels, final_values):
        """
        General function to plot profiles with P90, P50, and P10 lines and callouts.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        for profile in profiles:
            ax.plot(t, profile, color=profile_color, alpha=0.1)
        ax.plot(t, P90, color=colors[0], linewidth=2, label=f"P90 {labels}")
        ax.plot(t, p50, color=colors[1], linewidth=2, label=f"P50 {labels}")
        ax.plot(t, P10, color=colors[2], linewidth=2, label=f"P10 {labels}")

        # Adding callouts
        ax.text(t[-1], P90[-1], f"{P90[-1]:.2f}", color=colors[0], fontsize=10,
                bbox=dict(facecolor="white", edgecolor=colors[0], boxstyle="round,pad=0.5"))
        ax.text(t[-1], p50[-1], f"{p50[-1]:.2f}", color=colors[1], fontsize=10,
                bbox=dict(facecolor="white", edgecolor=colors[1], boxstyle="round,pad=0.5"))
        ax.text(t[-1], P10[-1], f"{P10[-1]:.2f}", color=colors[2], fontsize=10,
                bbox=dict(facecolor="white", edgecolor=colors[2], boxstyle="round,pad=0.5"))

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid()
        st.pyplot(fig)
        plt.close(fig)


    # Plot cumulative production with callouts
    st.header(cum_label)  # Add a header for cumulative production
    plot_with_callouts(
        t,
        cumulative_productions,
        P90_cum, P50_cum, P10_cum,
        f"Cumulative {title_suffix} Production (P90, P50, P10)",
        "Time (years)",
        f"Cumulative Production ({'MMSTB' if is_oil_field else 'Bcf'})",
        ["red", "black", "purple"],
        "Cumulative",
        [P90_cum[-1], P50_cum[-1], P10_cum[-1]]
    )
    
    # Plot incremental production with callouts
    st.header(incr_label)  # Add a header for incremental production
    plot_with_callouts(
        t,
        incremental_productions,
        P90_incr, P50_incr, P10_incr,
        f"Incremental {title_suffix} Production (P90, P50, P10)",
        "Time (years)",
        f"Incremental Production ({'MMSTB' if is_oil_field else 'Bcf'})",
        ["red", "black", "purple"],
        "Incremental",
        [P90_incr[-1], P50_incr[-1], P10_incr[-1]]
    )




#----------------------------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------------------------
    # Function to calculate annualized production rates
    def annualize_production(profile):
        if profile is None or len(profile) == 0:
            return []
        months_per_year = 12
        annualized = [
            np.sum(profile[i:i + months_per_year]) / min(months_per_year, len(profile) - i)
            for i in range(0, len(profile), months_per_year)
        ]
        return annualized
    
    
    
        # Function to get year-end values (rate / cumulative / incremental) Added on Dec 2025
    def get_year_end_profile(profile, t):
        """
        Return profile value at the end of each full year
        using the last time step <= integer year.
        t must be in years (same as used to build the profiles).
        """
        if profile is None or len(profile) == 0:
            return []
    
        # Number of full years covered by the simulation
        n_years = int(np.floor(t[-1]))
        values = []
    
        for yr in range(1, n_years + 1):
            # index of last time step within that year
            idx = np.searchsorted(t, yr, side="right") - 1
            if idx < 0:
                idx = 0
            values.append(profile[idx])
    
        return values


#----------------------------------------------------------------------------------------------------------    
    
    
    
    # Check and prepare data for all field types
    if field_type in ["Oil Field", "Gas Field", "Oil Field with WI", "Oil Field with WI & GI"]:
        st.header(f"{field_type}: Annualized Production")

        # Check if production profiles exist
        missing_profiles = []
        required_profiles = [
            "P90_prod", "P50_prod", "P10_prod",
            "P90_cum", "P50_cum", "P10_cum",
            "P90_incr", "P50_incr", "P10_incr"
        ]

        for profile_name in required_profiles:
            if profile_name not in locals() or eval(profile_name) is None:
                missing_profiles.append(profile_name)

        if missing_profiles:
            st.error(
                f"Missing production profiles: {', '.join(missing_profiles)}. "
                "Please run the simulation first."
            )
        else:
            # ------------------------------
            # ANNUALIZED (AVERAGE) PRODUCTION
            # ------------------------------

            # Calculate Annualized Rates, Cumulative, and Incremental Production
            P90_annual_rate = annualize_production(P90_prod)
            P50_annual_rate = annualize_production(P50_prod)
            P10_annual_rate = annualize_production(P10_prod)

            P90_cumulative = annualize_production(P90_cum)
            P50_cumulative = annualize_production(P50_cum)
            P10_cumulative = annualize_production(P10_cum)

            P90_incremental = annualize_production(P90_incr)
            P50_incremental = annualize_production(P50_incr)
            P10_incremental = annualize_production(P10_incr)

            # Ensure alignment of years with annualized rates
            years = np.arange(1, len(P90_annual_rate) + 1)

            # Set labels and units based on field type
            if field_type == "Gas Field":
                y_label_main = "Total Gas Rate (MMscf/d)"
                y_label_secondary = "Incremental Gas (Bcf)"
                table_unit_main = "MMscf/d"
                table_unit_secondary = "Bcf"
            else:
                y_label_main = "Total Oil Rate (Mstb/d)"
                y_label_secondary = "Incremental Oil (MMSTB)"
                table_unit_main = "Mstb/d"
                table_unit_secondary = "MMSTB"

            # Plot the data (annualized)
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # Bar chart for Annualized Rates
            bar_width = 0.25
            ax1.bar(years - bar_width, P90_annual_rate, width=bar_width,
                    label="P90 Rate", color='red', alpha=0.7)
            ax1.bar(years, P50_annual_rate, width=bar_width,
                    label="P50 Rate", color='black', alpha=0.7)
            ax1.bar(years + bar_width, P10_annual_rate, width=bar_width,
                    label="P10 Rate", color='purple', alpha=0.7)

            # Primary y-axis
            ax1.set_xlabel("Years")
            ax1.set_ylabel(y_label_main, color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')

            # Line plot for Incremental (Secondary Y-Axis)
            ax2 = ax1.twinx()
            ax2.plot(years, P90_incremental[:len(years)], "r--", linewidth=2, label="P90 Incremental")
            ax2.plot(years, P50_incremental[:len(years)], "k--", linewidth=2, label="P50 Incremental")
            ax2.plot(years, P10_incremental[:len(years)], "purple", linestyle="--", linewidth=2, label="P10 Incremental")

            # Secondary y-axis label
            ax2.set_ylabel(y_label_secondary, color='green')
            ax2.tick_params(axis='y', labelcolor='green')

            # Title and Grid
            fig.suptitle(f"{field_type}: Annualized Rates and Incremental Production")
            fig.tight_layout()
            ax1.grid(True)

            # Combine legends from both axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2,
                       loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

            # Display the plot in Streamlit
            st.pyplot(fig)
            plt.close(fig)

            # Prepare and Display Annualized Table
            data = {
                "Year": years,
                f"P90 Annualized Rate ({table_unit_main})": P90_annual_rate,
                f"P50 Annualized Rate ({table_unit_main})": P50_annual_rate,
                f"P10 Annualized Rate ({table_unit_main})": P10_annual_rate,
                f"P90 Cumulative ({table_unit_secondary})": P90_cumulative,
                f"P50 Cumulative ({table_unit_secondary})": P50_cumulative,
                f"P10 Cumulative ({table_unit_secondary})": P10_cumulative,
                f"P90 Incremental ({table_unit_secondary})": P90_incremental,
                f"P50 Incremental ({table_unit_secondary})": P50_incremental,
                f"P10 Incremental ({table_unit_secondary})": P10_incremental,
            }

            output_df = pd.DataFrame(data)

            st.header(f"{field_type}: Annualized Production Table")
            st.dataframe(
                output_df.style.format({
                    f"P90 Annualized Rate ({table_unit_main})": "{:.2f}",
                    f"P50 Annualized Rate ({table_unit_main})": "{:.2f}",
                    f"P10 Annualized Rate ({table_unit_main})": "{:.2f}",
                    f"P90 Cumulative ({table_unit_secondary})": "{:.2f}",
                    f"P50 Cumulative ({table_unit_secondary})": "{:.2f}",
                    f"P10 Cumulative ({table_unit_secondary})": "{:.2f}",
                    f"P90 Incremental ({table_unit_secondary})": "{:.2f}",
                    f"P50 Incremental ({table_unit_secondary})": "{:.2f}",
                    f"P10 Incremental ({table_unit_secondary})": "{:.2f}",
                })
            )

            # ---------------------------------------------
            # YEAR-END (LAST DAY OF EACH YEAR) PRODUCTION
            # ---------------------------------------------

            # Rate, Cumulative and Incremental at year-end
            P90_rate_year_end = get_year_end_profile(P90_prod, t)
            P50_rate_year_end = get_year_end_profile(P50_prod, t)
            P10_rate_year_end = get_year_end_profile(P10_prod, t)

            P90_cum_year_end = get_year_end_profile(P90_cum, t)
            P50_cum_year_end = get_year_end_profile(P50_cum, t)
            P10_cum_year_end = get_year_end_profile(P10_cum, t)

            P90_incr_year_end = get_year_end_profile(P90_incr, t)
            P50_incr_year_end = get_year_end_profile(P50_incr, t)
            P10_incr_year_end = get_year_end_profile(P10_incr, t)

            years_ye = np.arange(1, len(P90_rate_year_end) + 1)

            year_end_data = {
                "Year": years_ye,
                f"P90 Year-End Rate ({table_unit_main})": P90_rate_year_end,
                f"P50 Year-End Rate ({table_unit_main})": P50_rate_year_end,
                f"P10 Year-End Rate ({table_unit_main})": P10_rate_year_end,
                f"P90 Year-End Cumulative ({table_unit_secondary})": P90_cum_year_end,
                f"P50 Year-End Cumulative ({table_unit_secondary})": P50_cum_year_end,
                f"P10 Year-End Cumulative ({table_unit_secondary})": P10_cum_year_end,
                f"P90 Year-End Incremental ({table_unit_secondary})": P90_incr_year_end,
                f"P50 Year-End Incremental ({table_unit_secondary})": P50_incr_year_end,
                f"P10 Year-End Incremental ({table_unit_secondary})": P10_incr_year_end,
            }

            year_end_df = pd.DataFrame(year_end_data)

            st.header(f"{field_type}: Year-End Production Table (Last Day of Year)")
            st.dataframe(
                year_end_df.style.format({
                    f"P90 Year-End Rate ({table_unit_main})": "{:.2f}",
                    f"P50 Year-End Rate ({table_unit_main})": "{:.2f}",
                    f"P10 Year-End Rate ({table_unit_main})": "{:.2f}",
                    f"P90 Year-End Cumulative ({table_unit_secondary})": "{:.2f}",
                    f"P50 Year-End Cumulative ({table_unit_secondary})": "{:.2f}",
                    f"P10 Year-End Cumulative ({table_unit_secondary})": "{:.2f}",
                    f"P90 Year-End Incremental ({table_unit_secondary})": "{:.2f}",
                    f"P50 Year-End Incremental ({table_unit_secondary})": "{:.2f}",
                    f"P10 Year-End Incremental ({table_unit_secondary})": "{:.2f}",
                })
            )

    else:
        st.warning("Production data is missing. Ensure simulations are run to generate production profiles.")

#----------------------------------------------------------------------------------------------------------
    # Function to plot active well count over time
    def plot_active_well_count(active_well_counts, P90_wells, P50_wells, P10_wells, title, max_slots):
        """
        Plots the active well count over time, showing cloud and P90, P50, P10 curves.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot each scenario as a cloud
        for scenario in active_well_counts:
            ax.plot(t, scenario, color=profile_color, alpha=0.1)

        # Plot P90, P50, P10 curves
        ax.plot(t, P90_wells, color="red", linewidth=2, label="P90 Active Wells")
        ax.plot(t, P50_wells, color="black", linewidth=2, label="P50 Active Wells")
        ax.plot(t, P10_wells, color="purple", linewidth=2, label="P10 Active Wells")

        # Title, labels, and legend
        ax.set_title(f"{title} (Slot Constraint: {max_slots})")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Active Wells")
        ax.legend()
        ax.grid()
        st.pyplot(fig)
        plt.close(fig)

    # Call the function to plot Active Well Count
    st.header("Active Well Count Over Time")
    plot_active_well_count(
        active_well_counts,
        P90_wells,
        P50_wells,
        P10_wells,
        "Active Well Count Over Time",
        max_slots
    )

#----------------------------------------------------------------------------------------------------------
    if field_type in ["Oil Field with WI", "Oil Field with WI & GI"]:
        # Function to plot active well count over time for Oil Field with WI
        def plot_active_well_count_with_WI(active_well_counts, producer_well_counts, injector_well_counts, P90_wells, P50_wells, P10_wells, title, max_slots):
            """
            Plots the active well count over time for Oil Field with WI, showing cloud, P90, P50, P10 curves,
            and differentiating between producers and injectors.
            """
            fig, ax = plt.subplots(figsize=(10, 6))
        
            # Plot each scenario as a cloud for total active wells
            for scenario in active_well_counts:
                ax.plot(t, scenario, color="lightblue", alpha=0.1, label="_nolegend_")
        
            # Plot separate clouds for producers and injectors
            for producer_scenario, injector_scenario in zip(producer_well_counts, injector_well_counts):
                ax.plot(t, producer_scenario, color="green", alpha=0.05, label="_nolegend_")
                ax.plot(t, injector_scenario, color="blue", alpha=0.05, label="_nolegend_")
        
            # Plot P90, P50, P10 curves for total active wells
            ax.plot(t, P90_wells, color="red", linewidth=2, label="P90 Active Wells (Total)")
            ax.plot(t, P50_wells, color="black", linewidth=2, label="P50 Active Wells (Total)")
            ax.plot(t, P10_wells, color="purple", linewidth=2, label="P10 Active Wells (Total)")
        
            # Plot P50 curves separately for producers and injectors
            P50_producers = np.percentile(np.array(producer_well_counts), 50, axis=0)
            P50_injectors = np.percentile(np.array(injector_well_counts), 50, axis=0)
        
            ax.plot(t, P50_producers, color="green", linewidth=2, linestyle="--", label="P50 Producers")
            ax.plot(t, P50_injectors, color="blue", linewidth=2, linestyle="--", label="P50 Injectors")
        
            # Title, labels, and legend
            ax.set_title(f"{title} (Slot Constraint: {max_slots})")
            ax.set_xlabel("Time (years)")
            ax.set_ylabel("Active Wells")
            ax.legend(loc="upper right")
            ax.grid()
            st.pyplot(fig)
            plt.close(fig)
        
        # Calculate P90, P50, P10 for active wells (total, producers, and injectors)
        active_well_counts = np.array(active_well_counts)
    
        
        producer_well_counts = [
            [len([well for well in well_schedule if well["start"] <= time and well["end"] >= time and str(well["well"]).startswith("P")]) for time in t]
            for well_schedule in well_schedules
        ]
        
        injector_well_counts = [
            [len([well for well in well_schedule if well["start"] <= time and well["end"] >= time and str(well["well"]).startswith("WI")]) for time in t]
            for well_schedule in well_schedules
        ]
        
        # Call the function to plot Active Well Count for Oil Field with WI
        st.header("Active Well Count Over Time (Producers and Injectors)")
        plot_active_well_count_with_WI(active_well_counts, producer_well_counts, injector_well_counts, P90_wells, P50_wells, P10_wells, "Active Well Count Over Time", max_slots)

#----------------------------------------------------------------------------------------------------------






    # Identify scenario indices for P90, P50, and P10 based on incremental oil production
    def find_incremental_percentile_indices(incremental_productions, P90_incr, P50_incr, P10_incr):
        """
        Identify actual scenarios closest to the final P90/P50/P10 incremental volumes.
        """
        final_values = np.asarray([profile[-1] for profile in incremental_productions], dtype=float)
        P90_index = int(np.argmin(np.abs(final_values - P90_incr[-1])))
        P50_index = int(np.argmin(np.abs(final_values - P50_incr[-1])))
        P10_index = int(np.argmin(np.abs(final_values - P10_incr[-1])))
        return P90_index, P50_index, P10_index
    
    # Calculate P90, P50, P10 indices based on incremental oil
    P90_index, P50_index, P10_index = find_incremental_percentile_indices(incremental_productions, P90_incr, P50_incr, P10_incr)
   
    # Function to plot well-by-well production for a single scenario
    def plot_well_production(well_productions, title, rate_label):
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, production in enumerate(well_productions):
            ax.plot(t, production, label=f"P{i+1}")
        ax.set_title(title)
        ax.set_xlabel("Time (years)")
        ax.set_ylabel(rate_label)
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=6
        )  # Legend at bottom with horizontal arrangement
        ax.grid()
        st.pyplot(fig)
        plt.close(fig)
    
    # Plotting P90, P50, P10 well-by-well production based on incremental production
    st.header("Well-by-Well Production")
    
    # Adjust titles and labels dynamically based on field type
    plot_well_production(all_well_productions[P90_index], f"P90 Well-by-Well {title_suffix} Rate", rate_label)
    plot_well_production(all_well_productions[P50_index], f"P50 Well-by-Well {title_suffix} Rate", rate_label)
    plot_well_production(all_well_productions[P10_index], f"P10 Well-by-Well {title_suffix} Rate", rate_label)


    # Gantt chart plotting function
    def plot_gantt_chart(well_schedule, title):
        fig, ax = plt.subplots(figsize=(10, 6))
        for entry in well_schedule:
            start = project_start_date + timedelta(days=entry["start"] * 365.25)
            end = project_start_date + timedelta(days=entry["end"] * 365.25)
            ax.barh(entry["well"], (end - start).days, left=mdates.date2num(start))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xlabel('Time')
        plt.ylabel('Wells')
        plt.title(title)
        plt.grid(True)
        st.pyplot(fig)
        plt.close(fig)
    
    # Plot Gantt charts for P90, P50, P10 based on incremental oil
    st.subheader("P90 Gantt Chart")
    plot_gantt_chart(well_schedules[P90_index], "P90 Well Schedule")
    
    st.subheader("P50 Gantt Chart")
    plot_gantt_chart(well_schedules[P50_index], "P50 Well Schedule")
    
    st.subheader("P10 Gantt Chart")
    plot_gantt_chart(well_schedules[P10_index], "P10 Well Schedule")
    
    # Function to plot histogram with percentiles and include percentiles in legend
    def plot_histogram(data, P90, p50, P10, title, xlabel):
        """
        Plot histogram with percentiles (P90, P50, P10) included in the legend.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(data, bins=30, alpha=0.7, color=profile_color, label="Data")  # Ensure single dataset
        ax.axvline(P90, color="red", linestyle="--", label=f"P90: {P90:.2f}")
        ax.axvline(p50, color="green", linestyle="--", label=f"P50: {p50:.2f}")
        ax.axvline(P10, color="purple", linestyle="--", label=f"P10: {P10:.2f}")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)



    # Prepare data for Initial Rate Histogram
    initial_rates = [
        well[np.argmax(well > 0)]  # Get the first non-zero production rate for each well
        for scenario_well_productions in all_well_productions
        for well in scenario_well_productions
        if np.any(well > 0)  # Ensure well has production
    ]
    
    # Calculate percentiles for Initial Rate
    if initial_rates:
        init_rate_P90, init_rate_p50, init_rate_P10 = np.percentile(initial_rates, [10, 50, 90])
    else:
        init_rate_P90, init_rate_p50, init_rate_P10 = 0, 0, 0
    
    # Plot Initial Rate Histogram
    st.header(f"Initial {title_suffix} Rate Histogram")
    plot_histogram(
        initial_rates,
        init_rate_P90,
        init_rate_p50,
        init_rate_P10,
        f"Initial {title_suffix} Rate Distribution",
        f"Initial {title_suffix} Rate ({'Mstb/d' if is_oil_field else 'MMscf/d'})",
    )
    
    # Calculate EUR per well (Incremental production at the tail end of the simulation for each well)
    EUR_per_well = [
        np.sum(well * 30.4375 * scenario_data["incremental_factor"]) / 1000
        for scenario_data, scenario_well_productions in zip(scenarios, all_well_productions)
        for well in scenario_well_productions
        if np.sum(well) > 0
    ]
    
    # Calculate percentiles for EUR per well
    if EUR_per_well:
        EUR_P90, EUR_p50, EUR_P10 = np.percentile(EUR_per_well, [10, 50, 90])
    else:
        EUR_P90, EUR_p50, EUR_P10 = 0, 0, 0
    
    # Plot EUR per Well Histogram
    st.header(f"EUR per Well Histogram ({title_suffix})")
    plot_histogram(
        EUR_per_well,
        EUR_P90,
        EUR_p50,
        EUR_P10,
        f"EUR per Well Distribution ({title_suffix})",
        f"EUR per Well ({'MMSTB' if is_oil_field else 'Bcf'})",
    )

    #----------------------------------------------------------------------------------------------------------
    # Function to load and parse type curves
    # Initialize the type_curve variable
    if field_type in ["Oil Field", "Oil Field with WI", "Oil Field with WI & GI"]:
        type_curve = None
        
        # Function to load and parse type curves
        def load_type_curve(uploaded_file):
            if uploaded_file is None:
                return None

            type_curve = pd.read_csv(uploaded_file)
            required_columns = ["Normalized Cum Oil, MMstb"]
            if calculate_water_cut:
                required_columns.append("Water Cut, fr")
            if calculate_gor:
                required_columns.append("GOR, scf/stb")

            missing = [c for c in required_columns if c not in type_curve.columns]
            if missing:
                st.error(f"Invalid CSV format. Missing column(s): {', '.join(missing)}")
                return None

            return type_curve
        
        # Load the oil type curve only when Water Cut and/or GOR is requested.
        if (calculate_water_cut or calculate_gor) and uploaded_csv is not None:
            type_curve = load_type_curve(uploaded_csv)
            if type_curve is None:
                st.error("Oil type-curve CSV could not be loaded. Please check the required columns.")
        elif calculate_water_cut or calculate_gor:
            st.warning("Please upload a valid oil type-curve CSV for the selected Water Cut/GOR calculation.")

        # Function to normalize cumulative oil
        def normalize_cumulative_oil(cumulative_productions):
            cumulative_productions = np.asarray(cumulative_productions, dtype=float)
            denom = np.max(cumulative_productions, axis=1, keepdims=True)
            denom = np.where(denom > 0, denom, 1.0)
            return cumulative_productions / denom
        
        # Function to calculate Water Cut based on normalized cumulative oil and type curve
        def calculate_water_cut_data(normalized_cum_oil, type_curve):
            water_cut_results = []
            if type_curve is not None:
                for scenario in normalized_cum_oil:
                    interpolated_wc = np.interp(
                        scenario, type_curve["Normalized Cum Oil, MMstb"], type_curve["Water Cut, fr"], left=0, right=1)
                    water_cut_results.append(interpolated_wc)
            return np.array(water_cut_results)
        
        # Function to calculate GOR based on normalized cumulative oil and type curve
        def calculate_gor_data(normalized_cum_oil, type_curve):
            gor_results = []
            if type_curve is not None:
                for scenario in normalized_cum_oil:
                    interpolated_gor = np.interp(
                        scenario, type_curve["Normalized Cum Oil, MMstb"], type_curve["GOR, scf/stb"], left=0, right=np.max(type_curve["GOR, scf/stb"]))
                    gor_results.append(interpolated_gor)
            return np.array(gor_results)
        
        # Function to calculate Total Water Rate
        def calculate_water_rate(production_profiles, water_cut_profiles):
            total_water_rate = []
            for oil_rate, water_cut in zip(production_profiles, water_cut_profiles):
                # Cap water cut at 0.99 to avoid division by zero
                water_cut = np.clip(water_cut, 0, 0.99)
                water_rate = oil_rate * (water_cut / (1 - water_cut))
                total_water_rate.append(water_rate)
            return np.array(total_water_rate)
        
        # Function to calculate Total Gas Rate based on GOR
        def calculate_gas_rate(production_profiles, gor_results):
            return production_profiles * gor_results / 1000
        
        # Updated plotting function to visualize Water Cut
        def plot_type_curve_wcut(results, P90, p50, P10, title, ylabel):
            fig, ax = plt.subplots(figsize=(10, 6))
            for profile in results:
                ax.plot(t, profile, color='lightblue', alpha=0.1)
            ax.plot(t, P90, color="red", linestyle="--", linewidth=2, label=f"P90")
            ax.plot(t, p50, color="green", linestyle="--", linewidth=2, label=f"P50")
            ax.plot(t, P10, color="purple", linestyle="--", linewidth=2, label=f"P10")
            ax.set_title(title)
            ax.set_xlabel("Time (years)")
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid()
            st.pyplot(fig)
            plt.close(fig)
        # Updated plotting function to visualizeGOR
        def plot_type_curve_gor(results, P90, p50, P10, title, ylabel):
            fig, ax = plt.subplots(figsize=(10, 6))
            for profile in results:
                ax.plot(t, profile, color='red', alpha=0.1)
            ax.plot(t, P90, color="red", linestyle="--", linewidth=2, label=f"P90")
            ax.plot(t, p50, color="green", linestyle="--", linewidth=2, label=f"P50")
            ax.plot(t, P10, color="purple", linestyle="--", linewidth=2, label=f"P10")
            ax.set_title(title)
            ax.set_xlabel("Time (years)")
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid()
            st.pyplot(fig)  
            plt.close(fig)
            
        
        # Plot Water Rate
        def plot_water_rate(t, profiles, P90, p50, P10, title, ylabel):
            fig, ax = plt.subplots(figsize=(10, 6))
            for profile in profiles:
                ax.plot(t, profile, color='lightblue', alpha=0.1)
            ax.plot(t, P90, color="red", linestyle="--", linewidth=2, label="P90")
            ax.plot(t, p50, color="green", linestyle="--", linewidth=2, label="P50")
            ax.plot(t, P10, color="purple", linestyle="--", linewidth=2, label="P10")
            ax.set_title(title)
            ax.set_xlabel("Time (years)")
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid()
            st.pyplot(fig)
            plt.close(fig)
            
        # Plot Gas Rate
        def plot_gas_rate(t, profiles, P90, p50, P10, title, ylabel):
            fig, ax = plt.subplots(figsize=(10, 6))
            for profile in profiles:
                ax.plot(t, profile, color="red", alpha=0.1)
            ax.plot(t, P90, color="red", linestyle="--", linewidth=2, label="P90")
            ax.plot(t, p50, color="green", linestyle="--", linewidth=2, label="P50")
            ax.plot(t, P10, color="purple", linestyle="--", linewidth=2, label="P10")
            ax.set_title(title)
            ax.set_xlabel("Time (years)")
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid()
            st.pyplot(fig)
            plt.close(fig)
        
        
        # Perform Water Cut Calculation
        if calculate_water_cut and type_curve is not None:
            st.header("Water Cut Results")
            normalized_cum_oil = normalize_cumulative_oil(cumulative_productions)
            water_cut_results = calculate_water_cut_data(normalized_cum_oil, type_curve)
            P90_wc, P50_wc, P10_wc = np.percentile(water_cut_results, [10, 50, 90], axis=0)
            plot_type_curve_wcut(
                water_cut_results, P90_wc, P50_wc, P10_wc, "Water Cut over Time", "Water Cut (Fraction)")
            # Calculate Total Water Rate
            water_rate_profiles = calculate_water_rate(np.array(production_profiles), water_cut_results)
            P90_water, P50_water, P10_water = np.percentile(water_rate_profiles, [10, 50, 90], axis=0)
            st.subheader("Total Water Rate (Mstb/d)")
            plot_water_rate(t, water_rate_profiles, P90_water, P50_water, P10_water, "Total Water Rate over Time", "Water Rate (Mstb/d)")
        
        # Perform GOR Calculation
        if calculate_gor and type_curve is not None:
            st.header("GOR Results")
            normalized_cum_oil = normalize_cumulative_oil(cumulative_productions)
            gor_results = calculate_gor_data(normalized_cum_oil, type_curve)
            P90_gor, P50_gor, P10_gor = np.percentile(gor_results, [10, 50, 90], axis=0)
            plot_type_curve_gor(
                gor_results, P90_gor, P50_gor, P10_gor, "GOR over Time", "GOR (scf/stb)")
            # Calculate Total Gas Rate
            gas_rate_profiles = calculate_gas_rate(np.array(production_profiles), gor_results)
            P90_gas, P50_gas, P10_gas = np.percentile(gas_rate_profiles, [10, 50, 90], axis=0)
            # Plot Total Gas Rate
            st.subheader("Total Gas Rate (MMscf/d)")
            plot_gas_rate(t, gas_rate_profiles, P90_gas, P50_gas, P10_gas, "Total Gas Rate over Time", "Gas Rate (MMscf/d)")
        
        # Generate and Display Final Table with Combined Results
        if calculate_water_cut and calculate_gor and type_curve is not None:
            st.header("Annualized Production Table with Water Rate, Gas Rate, Water Cut, and GOR (P90, P50)")
        
            # Step 1: Annualize Water, Gas, and Oil Production for all scenarios
            P90_annual_water_rate = annualize_production(P90_water)  # P90 Water Rate
            P50_annual_water_rate = annualize_production(P50_water)  # P50 Water Rate
        
            P90_annual_gas_rate = annualize_production(P90_gas)  # P90 Gas Rate
            P50_annual_gas_rate = annualize_production(P50_gas)  # P50 Gas Rate
        
            P90_annual_oil_rate = annualize_production(P90_prod)  # P90 Oil Rate
            P50_annual_oil_rate = annualize_production(P50_prod)  # P50 Oil Rate
        
            # Step 2: Calculate Water Cut and GOR
            P90_water_cut = np.array(P90_annual_water_rate) / (np.array(P90_annual_oil_rate) + np.array(P90_annual_water_rate))
            P50_water_cut = np.array(P50_annual_water_rate) / (np.array(P50_annual_oil_rate) + np.array(P50_annual_water_rate))
        
            P90_gor = np.array(P90_annual_gas_rate) / np.array(P90_annual_oil_rate) * 1000
            P50_gor = np.array(P50_annual_gas_rate) / np.array(P50_annual_oil_rate) * 1000
        
            # Clean NaN and divide-by-zero errors
            P90_water_cut = np.nan_to_num(P90_water_cut, nan=0.0)
            P50_water_cut = np.nan_to_num(P50_water_cut, nan=0.0)
            P90_gor = np.nan_to_num(P90_gor, nan=0.0)
            P50_gor = np.nan_to_num(P50_gor, nan=0.0)
        
            # Step 3: Prepare Table Data
            table_data = pd.DataFrame({
                "Year": np.arange(1, len(P90_annual_water_rate) + 1),
                "P90 Annualized Water Rate (Mstb/d)": P90_annual_water_rate,
                "P50 Annualized Water Rate (Mstb/d)": P50_annual_water_rate,
                "P90 Annualized Gas Rate (MMscf/d)": P90_annual_gas_rate,
                "P50 Annualized Gas Rate (MMscf/d)": P50_annual_gas_rate,
                "P90 Water Cut (Fraction)": P90_water_cut,
                "P50 Water Cut (Fraction)": P50_water_cut,
                "P90 GOR (scf/stb)": P90_gor,
                "P50 GOR (scf/stb)": P50_gor
            })
        
            # Step 4: Display Table in Streamlit
            st.dataframe(table_data.style.format({
                "P90 Annualized Water Rate (Mstb/d)": "{:.2f}",
                "P50 Annualized Water Rate (Mstb/d)": "{:.2f}",
                "P90 Annualized Gas Rate (MMscf/d)": "{:.2f}",
                "P50 Annualized Gas Rate (MMscf/d)": "{:.2f}",
                "P90 Water Cut (Fraction)": "{:.2f}",
                "P50 Water Cut (Fraction)": "{:.2f}",
                "P90 GOR (scf/stb)": "{:.0f}",
                "P50 GOR (scf/stb)": "{:.0f}"
            }))


# TORNADO for all field type
#-------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------

    def calculate_incremental_production(base_params, changed_param=None, changed_value=None):
        """
        Simulates the incremental production based on input parameters.
        Optionally changes one parameter to calculate sensitivity.
        """
        params = base_params.copy()
        if changed_param and changed_value is not None:
            params[changed_param] = changed_value
    
        # Calculate incremental production with realistic weighting for parameters
        production = (
            params["num_producers"]
            * params["initial_rate"]
            * params["incremental_factor"]
            * ((1 - params["yearly_decline"] / 100) ** (params.get("drilling_duration", 365.25) / 365.25))
            * (1 + 0.1 * params["b"])
        )
        if "water_injection_rate" in params:
            production *= (1 + 0.05 * params["water_injection_rate"])
        if "gas_injection_rate" in params:
            production *= (1 + 0.05 * params["gas_injection_rate"])
        return production
    
    # Sensitivity analysis function
    def sensitivity_analysis(base_params, sensitivity_ranges):
        results = {}
        base_incremental_production = calculate_incremental_production(base_params)
    
        for param, (low, high) in sensitivity_ranges.items():
            # Calculate downside and upside
            downside_production = calculate_incremental_production(base_params, param, low)
            downside_delta = downside_production - base_incremental_production
            upside_production = calculate_incremental_production(base_params, param, high)
            upside_delta = upside_production - base_incremental_production
    
            # Store results
            results[param] = {"downside": downside_delta, "upside": upside_delta}
        return results
    
    # Function to plot Tornado chart
    def plot_tornado_chart(results, title="Tornado Plot", unit_label="Incremental Production"):
        sorted_results = sorted(
            results.items(),
            key=lambda x: abs(x[1]["upside"]) + abs(x[1]["downside"]),
            reverse=False
        )
        params = [item[0] for item in sorted_results]
        downsides = [item[1]["downside"] for item in sorted_results]
        upsides = [item[1]["upside"] for item in sorted_results]
    
        fig, ax = plt.subplots(figsize=(10, 6))
        y_positions = np.arange(len(params))
        ax.barh(y_positions, upsides, color="green", label="Upside Case")
        ax.barh(y_positions, downsides, color="red", label="Downside Case")
        ax.axvline(0, color="black", linestyle="--", label="Baseline Incremental Production")
        ax.set_yticks(y_positions)
        ax.set_yticklabels(params)
        ax.set_xlabel(f"{unit_label} Difference")
        ax.set_title(title)
        ax.legend()
        return fig
    
    # Tornado Section Logic
    if calculate_tornado:
        st.subheader("Tornado Sensitivity Analysis")
        # field_type = st.selectbox("Select Field Type:", ["Oil Field", "Gas Field", "Oil Field with WI", "Oil Field with WI & GI"])
    
        # Field-specific configuration
        unit_label, rate_label, default_rate = {
            "Oil Field": ("Incremental Oil (MMSTB)", "Initial Oil Rate (Mstb/d)", 3.0),
            "Gas Field": ("Incremental Gas (Bcf)", "Initial Gas Rate (MMscf/d)", 30.0),
            "Oil Field with WI": ("Incremental Oil with Water Injection (MMSTB)", "Initial Oil Rate (Mstb/d)", 3.0),
            "Oil Field with WI & GI": ("Incremental Oil with Water and Gas Injection (MMSTB)", "Initial Oil Rate (Mstb/d)", 3.0)
        }[field_type]
    
        # Manual or auto base case input
        base_case_option = st.radio(
            "Select Base Case Input Method:",
            ("Use Mean of Input Parameters", "Custom Base Case Input"),
        )
    
        if base_case_option == "Use Mean of Input Parameters":
            if adjust_schedule:
                num_producers = sum(wells for year, wells in manual_schedule)
            base_params = {
                "num_producers": np.mean(num_producers) if not adjust_schedule else num_producers,
                "initial_rate": np.mean(initial_rate),
                "incremental_factor": np.mean(incremental_factor),
                "yearly_decline": np.mean(yearly_decline),
                "b": np.mean(arps_b),
            }
            if not adjust_schedule and drilling_duration is not None:
                base_params["drilling_duration"] = np.mean(drilling_duration)
        elif base_case_option == "Custom Base Case Input":
            base_params = {
                "num_producers": st.number_input("Input Base Case for Number of Producers:", value=30),
                "initial_rate": st.number_input(f"Input Base Case for {rate_label}:", value=default_rate),
                "incremental_factor": st.number_input("Input Base Case for Incremental Factor:", value=0.5),
                "yearly_decline": st.number_input("Input Base Case for Yearly Decline (%):", value=15.0),
                "b": st.number_input("Input Base Case for Arps Decline Parameter b:", value=0.3),
            }
            if not adjust_schedule:
                base_params["drilling_duration"] = st.number_input("Input Base Case for Drilling Duration (days):", value=100)
    
        # Sensitivity ranges, consider adding dynamic factors if applicable
        sensitivity_ranges = {
            "num_producers": (10, 50),
            "initial_rate": (default_rate / 2, default_rate * 2),
            "incremental_factor": (0.2, 1.0),
            "yearly_decline": (5, 30),
            "b": (0.1, 0.5),
        }
        if not adjust_schedule:
            sensitivity_ranges["drilling_duration"] = (30, 200)
    
        # Perform Sensitivity Analysis
        sensitivity_results = sensitivity_analysis(base_params, sensitivity_ranges)
    
        # Plot Tornado Chart
        fig = plot_tornado_chart(
            sensitivity_results,
            title=f"{field_type} Tornado Plot",
            unit_label=unit_label
        )
        st.pyplot(fig)
        plt.close(fig)

    # #------------------------------------------------------------------------------------------------------------------------------------  
    # E C O N O M I C S
    # -----------------------------------------------------------------------------------------------------------------------
    # Checkbox to toggle economics calculation
    # calculate_economics = st.sidebar.checkbox("Calculate Economics")
    
    if calculate_economics:
        # Title
        st.title("Economic Metrics and Analysis")
    
        # Input parameters
        st.sidebar.header("Input Parameters")
    
        # Dynamic labels and defaults based on field type
        if is_oil_field:
            price_label = "Oil Price (USD/bbl)"
            opex_label = "Operating Cost per Barrel (USD/bbl)"
            unit_label = "MMSTB"
            rate_label = "Mstb/d"
            default_price = 70.0
            default_opex = 10.0
        else:  # Gas Field
            price_label = "Gas Price (USD/Mscf)"
            opex_label = "Operating Cost per Mscf (USD/Mscf)"
            unit_label = "Bcf"
            rate_label = "MMscf/d"
            default_price = 3.0
            default_opex = 0.5
    
        product_price = st.sidebar.number_input(price_label, value=default_price)
        discount_rate = st.sidebar.number_input("Discount Rate (%)", value=10.0) / 100
        capex = st.sidebar.number_input("Total CAPEX (MM$)", value=1000.0)
        drilling_cost_per_well = st.sidebar.number_input("Drilling Cost per Well (MM$)", value=10.0)
        opex_per_unit = st.sidebar.number_input(opex_label, value=default_opex)
    
        # Function to calculate annualized production
        def annualize_production(profile):
            months_per_year = 12
            return [np.sum(profile[i:i + months_per_year]) / months_per_year for i in range(0, len(profile), months_per_year)]
    
        # Annualized production for P90, P50, P10
        P90_annual_rate = annualize_production(P90_prod)
        P50_annual_rate = annualize_production(P50_prod)
        P10_annual_rate = annualize_production(P10_prod)
    
        # Function to calculate wells per year
        def calculate_wells_per_year(well_schedule, simulation_years):
            wells_per_year = np.zeros(simulation_years, dtype=int)
            for well in well_schedule:
                start_year = int(well["start"])
                if start_year < simulation_years:
                    wells_per_year[start_year] += 1
            return wells_per_year
    
        simulation_years = max(len(P90_annual_rate), len(P50_annual_rate), len(P10_annual_rate))
        years = np.arange(1, simulation_years + 1)
    
        # Wells per year for P90, P50, P10
        P90_wells_per_year = calculate_wells_per_year(well_schedules[P90_index], simulation_years)
        P50_wells_per_year = calculate_wells_per_year(well_schedules[P50_index], simulation_years)
        P10_wells_per_year = calculate_wells_per_year(well_schedules[P10_index], simulation_years)
    
        # Function to calculate cashflows
        def calculate_cashflows(production, wells, capex, drilling_cost, opex, price, conversion_factor):
            cashflows = [-capex]
            for rate, num_wells in zip(production, wells):
                revenue = rate * price * 365 / conversion_factor
                opex_cost = rate * opex * 365 / conversion_factor
                cashflows.append(revenue - opex_cost - (num_wells * drilling_cost))
            return np.array(cashflows)
    
        # Conversion factor
        conversion_factor = 1000  # Applies to both Oil and Gas Fields
    
        # Cashflows for P90, P50, P10
        P90_cashflows = calculate_cashflows(P90_annual_rate, P90_wells_per_year, capex, drilling_cost_per_well, opex_per_unit, product_price, conversion_factor)
        P50_cashflows = calculate_cashflows(P50_annual_rate, P50_wells_per_year, capex, drilling_cost_per_well, opex_per_unit, product_price, conversion_factor)
        P10_cashflows = calculate_cashflows(P10_annual_rate, P10_wells_per_year, capex, drilling_cost_per_well, opex_per_unit, product_price, conversion_factor)
    
        # Function to calculate cumulative cashflows and payback periods
        def calculate_payback_periods(cashflows, discount_rate):
            years = np.arange(len(cashflows))
            cumulative_cashflows = np.cumsum(cashflows)
            discounted_cashflows = cashflows / (1 + discount_rate) ** years
            cumulative_discounted_cashflows = np.cumsum(discounted_cashflows)
    
            # Simple Payback Period (SPP)
            if np.any(cumulative_cashflows >= 0):
                spp_idx = np.argmax(cumulative_cashflows >= 0)
                if spp_idx > 0:
                    spp = spp_idx - 1 + (0 - cumulative_cashflows[spp_idx - 1]) / (
                        cumulative_cashflows[spp_idx] - cumulative_cashflows[spp_idx - 1]
                    )
                else:
                    spp = float(spp_idx)
            else:
                spp = None
    
            # Discounted Payback Period (DPP)
            if np.any(cumulative_discounted_cashflows >= 0):
                dpp_idx = np.argmax(cumulative_discounted_cashflows >= 0)
                if dpp_idx > 0:
                    dpp = dpp_idx - 1 + (0 - cumulative_discounted_cashflows[dpp_idx - 1]) / (
                        cumulative_discounted_cashflows[dpp_idx] - cumulative_discounted_cashflows[dpp_idx - 1]
                    )
                else:
                    dpp = float(dpp_idx)
            else:
                dpp = None
    
            return cumulative_cashflows, cumulative_discounted_cashflows, spp, dpp
    
        # Payback Periods for P90, P50, P10
        P90_cum_cash, P90_disc_cash, P90_spp, P90_dpp = calculate_payback_periods(P90_cashflows, discount_rate)
        P50_cum_cash, P50_disc_cash, P50_spp, P50_dpp = calculate_payback_periods(P50_cashflows, discount_rate)
        P10_cum_cash, P10_disc_cash, P10_spp, P10_dpp = calculate_payback_periods(P10_cashflows, discount_rate)
    
        # Function to plot Payback Periods
        def plot_payback_periods(years, cumulative_cashflows, discounted_cashflows, spp, dpp, title):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(years, cumulative_cashflows, label="Simple Payback (SPP)", color="blue")
            ax.plot(years, discounted_cashflows, label="Discounted Payback (DPP)", color="orange")
    
            if spp is not None:
                ax.axvline(spp, color="blue", linestyle="--", label=f"SPP: Year {spp:.1f}")
            if dpp is not None:
                ax.axvline(dpp, color="orange", linestyle="--", label=f"DPP: Year {dpp:.1f}")
    
            ax.set_title(title)
            ax.set_xlabel("Years")
            ax.set_ylabel("Cumulative Cashflows (MM$)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            plt.close(fig)
    
        # Plots for P90, P50, P10 Payback Periods
        st.subheader("P90 Payback Periods")
        plot_payback_periods(years, P90_cum_cash[1:], P90_disc_cash[1:], P90_spp, P90_dpp, "P90")
    
        st.subheader("P50 Payback Periods")
        plot_payback_periods(years, P50_cum_cash[1:], P50_disc_cash[1:], P50_spp, P50_dpp, "P50")
    
        st.subheader("P10 Payback Periods")
        plot_payback_periods(years, P10_cum_cash[1:], P10_disc_cash[1:], P10_spp, P10_dpp, "P10")
    
        # NPV Plot
        st.subheader("Cumulative NPV Over Time")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(years, P90_cum_cash[1:], label="P90 NPV")
        ax.plot(years, P50_cum_cash[1:], label="P50 NPV")
        ax.plot(years, P10_cum_cash[1:], label="P10 NPV")
        ax.set_xlabel("Years")
        ax.set_ylabel("Cumulative NPV (MM$)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)



    

        # Economic Summary Table
        summary_data = {
            "Metric": ["Total NPV (MM$)", "Discounted Payback Period (Years)", "Number of Wells Drilled"],
            "P90": [round(P90_cum_cash[-1], 2) if P90_cum_cash[-1] is not None else np.nan, P90_dpp if P90_dpp is not None else np.nan,int(sum(P90_wells_per_year))],
            "P50": [round(P50_cum_cash[-1], 2) if P50_cum_cash[-1] is not None else np.nan,P50_dpp if P50_dpp is not None else np.nan,int(sum(P50_wells_per_year))],
            "P10": [round(P10_cum_cash[-1], 2) if P10_cum_cash[-1] is not None else np.nan,P10_dpp if P10_dpp is not None else np.nan,int(sum(P10_wells_per_year))],
        }

        summary_df = pd.DataFrame(summary_data)
        st.table(summary_df)

        
        # Updated Yearly Metrics Table
        table_df = pd.DataFrame({
            "Years": years,
            f"P90 Total Rate ({rate_label})": P90_annual_rate,
            f"P50 Total Rate ({rate_label})": P50_annual_rate,
            f"P10 Total Rate ({rate_label})": P10_annual_rate,
            "P90 Number of Wells": P90_wells_per_year,
            "P50 Number of Wells": P50_wells_per_year,
            "P10 Number of Wells": P10_wells_per_year,
            "P90 NPV (MM$)": P90_cum_cash[1:],  # Adjusted to exclude the initial cashflow
            "P50 NPV (MM$)": P50_cum_cash[1:],
            "P10 NPV (MM$)": P10_cum_cash[1:]
        })
        st.dataframe(table_df)
        
        # Function to calculate number of wells per year and their start dates
        def calculate_wells_per_year(well_schedule, project_start_date, total_years):
            wells_per_year = [0] * total_years  # Initialize with zeros for each year
            start_dates = [[] for _ in range(total_years)]  # List of lists for each year's start dates
            
            for well in well_schedule:
                # Calculate the year in which the well starts production
                start_year = int(well["start"])
                if start_year < total_years:
                    wells_per_year[start_year] += 1
                    # Append the formatted start date
                    start_date = (project_start_date + timedelta(days=well["start"] * 365.25)).strftime("%Y-%m-%d")
                    start_dates[start_year].append(start_date)
            
            return wells_per_year, start_dates
        
        # Project start date
        # project_start_date = datetime(2029, 1, 1)
        
        # Calculate wells per year for P90, P50, and P10
        P90_wells_per_year, P90_start_dates = calculate_wells_per_year(well_schedules[P90_index], project_start_date, len(P90_annual_rate))
        P50_wells_per_year, P50_start_dates = calculate_wells_per_year(well_schedules[P50_index], project_start_date, len(P50_annual_rate))
        P10_wells_per_year, P10_start_dates = calculate_wells_per_year(well_schedules[P10_index], project_start_date, len(P10_annual_rate))
        
        # Determine the last year with drilled wells
        last_drilled_year = max(
            max(np.nonzero(P90_wells_per_year)[0], default=0),
            max(np.nonzero(P50_wells_per_year)[0], default=0),
            max(np.nonzero(P10_wells_per_year)[0], default=0),
        )
        
        # Truncate the data to only include years with wells drilled
        P90_wells_per_year = P90_wells_per_year[:last_drilled_year + 1]
        P50_wells_per_year = P50_wells_per_year[:last_drilled_year + 1]
        P10_wells_per_year = P10_wells_per_year[:last_drilled_year + 1]
        P90_start_dates = P90_start_dates[:last_drilled_year + 1]
        P50_start_dates = P50_start_dates[:last_drilled_year + 1]
        P10_start_dates = P10_start_dates[:last_drilled_year + 1]
        P90_annual_rate = P90_annual_rate[:last_drilled_year + 1]
        P50_annual_rate = P50_annual_rate[:last_drilled_year + 1]
        P10_annual_rate = P10_annual_rate[:last_drilled_year + 1]
        years = np.arange(1, last_drilled_year + 2)  # Adjust years accordingly
        
        # Prepare data for well details printout
        well_details_data = {
            "Years": years,
            "P90 Wells/Year": P90_wells_per_year,
            "P50 Wells/Year": P50_wells_per_year,
            "P10 Wells/Year": P10_wells_per_year,
            "P90 Start Dates": P90_start_dates,
            "P50 Start Dates": P50_start_dates,
            "P10 Start Dates": P10_start_dates,
        }
        
        # Convert to DataFrame
        well_details_df = pd.DataFrame(well_details_data)
        
        # Display the well details
        st.header("Well Details Per Year")
        st.write(
            "The table below shows the number of wells and their respective start dates for each year "
            "for P90, P50, and P10 cases."
        )
        st.dataframe(well_details_df)

        

#-------------------------------------------------------------------------------------------------------------------------------------    
    
    # S Curve Calculation and Plot
    # Extract tail-end cumulative incremental production for each simulation
    tail_end_incremental = [profile[-1] for profile in incremental_productions]
    
    # Sort the tail-end values to create S-Curve
    sorted_incremental = np.sort(tail_end_incremental)
    cumulative_prob = np.linspace(0, 100, len(sorted_incremental))
    
    # Calculate P90, P50, P10 from the tail-end values
    P90 = np.percentile(sorted_incremental, 10)
    p50 = np.percentile(sorted_incremental, 50)
    P10 = np.percentile(sorted_incremental, 90)
    
    # Identify indices of scenarios corresponding to P90, P50, P10
    P90_index = np.argmin(np.abs(tail_end_incremental - P90))
    P50_index = np.argmin(np.abs(tail_end_incremental - p50))
    P10_index = np.argmin(np.abs(tail_end_incremental - P10))
    
    # Use these scenarios for time-series plotting
    P90_incr = incremental_productions[P90_index]
    P50_incr = incremental_productions[P50_index]
    P10_incr = incremental_productions[P10_index]
    
    # S-Curve Plot
    fig_s, ax_s = plt.subplots(figsize=(10, 6))
    ax_s.plot(sorted_incremental, cumulative_prob, '--', linewidth=2, color='blue', label='S-Curve')
    ax_s.scatter([P90, p50, P10], [10, 50, 90], color='red', zorder=5)
    
    # Annotate P90, P50, P10
    for prob, value, label in zip([10, 50, 90], [P90, p50, P10], ["P90", "P50", "P10"]):
        ax_s.text(value, prob, f"{label}: {value:.2f} {'MMSTB' if is_oil_field else 'Bcf'}", ha='center', fontsize=10, color='red')
    
    ax_s.set_xlabel(f"Incremental {title_suffix} Production ({'MMSTB' if is_oil_field else 'Bcf'})")
    ax_s.set_ylabel("Cumulative Probability (%)")
    ax_s.set_title(f"Probabilistic S-Curve Distribution of Incremental {title_suffix}")
    ax_s.grid(True)
    ax_s.legend()
    st.pyplot(fig_s)
    plt.close(fig_s)
    
    # Time-Series Plot for Selected Scenarios
    fig_t, ax_t = plt.subplots(figsize=(10, 6))
    ax_t.plot(t, P90_incr, 'r', label=f'P90 Incremental ({P90:.2f} {"MMSTB" if is_oil_field else "Bcf"})')
    ax_t.plot(t, P50_incr, 'k', label=f'P50 Incremental ({p50:.2f} {"MMSTB" if is_oil_field else "Bcf"})')
    ax_t.plot(t, P10_incr, 'purple', label=f'P10 Incremental ({P10:.2f} {"MMSTB" if is_oil_field else "Bcf"})')
    
    ax_t.set_xlabel("Time (years)")
    ax_t.set_ylabel(f"Incremental Production ({'MMSTB' if is_oil_field else 'Bcf'})")
    ax_t.set_title(f"Incremental {title_suffix} Production (P90, P50, P10)")
    ax_t.legend()
    ax_t.grid(True)
    st.pyplot(fig_t)
    plt.close(fig_t)


    #----------------------------------------------------------------------------------------------------------

    # Cloud of Reservoir Pressure and Recovery Factor across simulations
    reservoir_pressure_cloud = []
    recovery_factor_cloud = []

    # STOIIP and Initial Pressure for each simulation
    if field_type == "Oil Field" or field_type == "Oil Field with WI" or field_type == "Oil Field with WI & GI":
        STOIIP_values = sample_range(stoiip_range, get_dist("dist_stoiip"), size=simulation_number)
        oil_in_place = STOIIP_values  # Use STOIIP for Oil Field calculations
    else:  # Gas Field
        STOIIP_values = sample_range(ogiip_range, get_dist("dist_ogiip"), size=simulation_number)
        oil_in_place = STOIIP_values  # Use OGIIP for Gas Field calculations

    # STOIIP_values = sample_range(stoiip_range, get_dist("dist_stoiip"), size=simulation_number)
    initial_pressure_values = sample_range(initial_pressure_range, get_dist("dist_initial_pressure"), size=simulation_number)

    # Compute reservoir pressure and recovery factor for all simulations
    for i in range(simulation_number):
        stoiip = STOIIP_values[i]
        initial_pressure = initial_pressure_values[i]
        cumulative_oil = cumulative_productions[i]  # Cumulative production profile for this simulation

        # Reservoir Pressure Decline
        reservoir_pressure = initial_pressure - (cumulative_oil / stoiip) * initial_pressure
        reservoir_pressure_cloud.append(reservoir_pressure)

        # Recovery Factor Progression
        recovery_factor = cumulative_oil / stoiip
        recovery_factor_cloud.append(recovery_factor)

    # Convert to numpy arrays
    reservoir_pressure_cloud = np.array(reservoir_pressure_cloud)
    recovery_factor_cloud = np.array(recovery_factor_cloud)

    # Calculate P90, P50, P10 for both pressure and recovery factor
    P90_pressure, P50_pressure, P10_pressure = compute_percentiles(reservoir_pressure_cloud)
    P90_rf, P50_rf, P10_rf = compute_percentiles(recovery_factor_cloud)

    # Plot Reservoir Pressure Cloud with P90, P50, and P10
    fig_pressure, ax_pressure = plt.subplots(figsize=(10, 6))
    for pressure in reservoir_pressure_cloud:
        ax_pressure.plot(t, pressure, color=profile_color, alpha=0.1)  # Pressure cloud
    ax_pressure.plot(t, P90_pressure, 'r', linewidth=2, label="P90 Pressure")
    ax_pressure.plot(t, P50_pressure, 'k', linewidth=2, label="P50 Pressure")
    ax_pressure.plot(t, P10_pressure, 'purple', linewidth=2, label="P10 Pressure")
    ax_pressure.set_xlabel("Time (years)")
    ax_pressure.set_ylabel("Reservoir Pressure (psia)")
    ax_pressure.set_title("Reservoir Pressure Decline Cloud with P90, P50, P10")
    ax_pressure.legend()
    ax_pressure.grid()
    st.pyplot(fig_pressure)
    plt.close(fig_pressure)  # Close figure to release memory

    # Plot Recovery Factor Cloud with P90, P50, and P10
    fig_rf, ax_rf = plt.subplots(figsize=(10, 6))
    for rf in recovery_factor_cloud:
        ax_rf.plot(t, rf, color=profile_color, alpha=0.1)  # Recovery factor cloud
    ax_rf.plot(t, P90_rf, 'r', linewidth=2, label="P90 Recovery Factor")
    ax_rf.plot(t, P50_rf, 'k', linewidth=2, label="P50 Recovery Factor")
    ax_rf.plot(t, P10_rf, 'purple', linewidth=2, label="P10 Recovery Factor")
    ax_rf.set_xlabel("Time (years)")
    ax_rf.set_ylabel("Recovery Factor (Fraction)")
    ax_rf.set_title("Recovery Factor Cloud with P90, P50, P10")
    ax_rf.legend()
    ax_rf.grid()
    st.pyplot(fig_rf)
    plt.close(fig_rf)  # Close figure to release memory



    # Generate a single STOIIP and Initial Pressure for calculations
    # Determine field-specific parameters
    if field_type == "Oil Field" or field_type == "Oil Field with WI" or field_type == "Oil Field with WI & GI" :
        STOIIP = np.mean(stoiip_range)  # Average STOIIP
        residual_fraction = residual_oil_fraction
        volume_label = "STOIIP"
        cumulative_label = "Cumulative Oil Production (MMSTB)"
    else:  # Gas Field
        STOIIP = np.mean(ogiip_range)  # Average OGIIP
        residual_fraction = 1.0 - gas_recovery_factor
        volume_label = "OGIIP"
        cumulative_label = "Cumulative Gas Production (Bcf)"
    
    # Calculate Reduced Extractable Volume
    extractable_volume = STOIIP * (1 - residual_fraction)
    
    # Calculate Total Cumulative Production for All Scenarios
    total_cumulative_production = [cumulative_production[-1] for cumulative_production in cumulative_productions]
    max_cum_production = np.max(total_cumulative_production)
    min_cum_production = np.min(total_cumulative_production)
    
    # Check if Total Cumulative Production is within Extractable Volume Constraints
    if max_cum_production > extractable_volume:
        st.error(
            f"Warning: Total cumulative production ({max_cum_production:.2f}) exceeds the extractable {volume_label} "
            f"({extractable_volume:.2f}). Consider increasing {volume_label} or reducing production."
        )
    elif min_cum_production < STOIIP * 0.1:
        st.warning(
            f"Warning: Total cumulative production ({min_cum_production:.2f}) is too low compared to {volume_label} "
            f"({STOIIP:.2f}). Check your input parameters for underproduction."
        )
    else:
        st.success(
            f"Total cumulative production is within the extractable {volume_label} limits ({extractable_volume:.2f})."
        )
    
    # Display Residual Volume (for Oil Field)
    if field_type == "Oil Field":
        st.subheader("Residual Oil Volume")
        residual_volume = STOIIP * residual_fraction
        st.write(f"**Residual Oil Remaining in Reservoir:** {residual_volume:.2f} MMSTB ({residual_fraction:.0%} of STOIIP)")
    
    # Plot Extractable Volume Cloud
    fig_volume, ax_volume = plt.subplots(figsize=(10, 6))
    
    # Plot cloud of cumulative production for all scenarios
    for i in range(simulation_number):
        ax_volume.plot(t, cumulative_productions[i], color=profile_color, alpha=0.1)
    
    # Overlay P90, P50, P10 cumulative production
    ax_volume.plot(t, P90_cum, color='red', linewidth=2, label=f'P90 Cumulative ({P90_cum[-1]:.2f})')
    ax_volume.plot(t, P50_cum, color='black', linewidth=2, label=f'P50 Cumulative ({P50_cum[-1]:.2f})')
    ax_volume.plot(t, P10_cum, color='purple', linewidth=2, label=f'P10 Cumulative ({P10_cum[-1]:.2f})')
    
    # Plot Extractable Volume as a horizontal line
    ax_volume.axhline(y=STOIIP, color='blue', linestyle='--', linewidth=2, label=f"{volume_label} ({STOIIP:.2f})")
    ax_volume.axhline(y=extractable_volume, color='green', linestyle='--', linewidth=2, 
                      label=f"Extractable {volume_label} ({extractable_volume:.2f})")
    
    # Labels and Titles
    ax_volume.set_title(f"{volume_label} Cloud with P90, P50, and P10 Cumulative Production")
    ax_volume.set_xlabel("Time (years)")
    ax_volume.set_ylabel(cumulative_label)
    ax_volume.legend()
    ax_volume.grid()
    
    # Display the plot in Streamlit
    st.pyplot(fig_volume)
    plt.close(fig_volume)




    # CGR WGR Section
# -------------------------------------------------------------------------------------------------------
    # Function to load and parse gas type curve
    def load_gas_type_curve(file):
        if file is None:
            return None

        df = pd.read_csv(file)
        required_columns = ["Normalized Cumulative Gas (Bcf)"]
        if calculate_cgr:
            required_columns.append("CGR (bbl/MMscf)")
        if calculate_wgr:
            required_columns.append("WGR (bbl/MMscf)")

        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            st.error(f"Invalid CSV format. Missing column(s): {', '.join(missing)}")
            return None
        return df
    
    # Function to normalize cumulative gas
    def normalize_cumulative_gas(cumulative_productions):
        cumulative_productions = np.asarray(cumulative_productions, dtype=float)
        denom = np.max(cumulative_productions, axis=1, keepdims=True)
        denom = np.where(denom > 0, denom, 1.0)
        return cumulative_productions / denom

    # Plot Water Rate
    def plot_water_rate(t, profiles, P90, p50, P10, title, ylabel):
        fig, ax = plt.subplots(figsize=(10, 6))
        for profile in profiles:
            ax.plot(t, profile, color='lightblue', alpha=0.1)
        ax.plot(t, P90, color="red", linestyle="--", linewidth=2, label="P90")
        ax.plot(t, p50, color="green", linestyle="--", linewidth=2, label="P50")
        ax.plot(t, P10, color="purple", linestyle="--", linewidth=2, label="P10")
        ax.set_title(title)
        ax.set_xlabel("Time (years)")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid()
        st.pyplot(fig)
        plt.close(fig)
        
    # Plot Gas Rate
    def plot_gas_rate(t, profiles, P90, p50, P10, title, ylabel):
        fig, ax = plt.subplots(figsize=(10, 6))
        for profile in profiles:
            ax.plot(t, profile, color='red', alpha=0.1)
        ax.plot(t, P90, color="red", linestyle="--", linewidth=2, label="P90")
        ax.plot(t, p50, color="green", linestyle="--", linewidth=2, label="P50")
        ax.plot(t, P10, color="purple", linestyle="--", linewidth=2, label="P10")
        ax.set_title(title)
        ax.set_xlabel("Time (years)")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid()
        st.pyplot(fig)
        plt.close(fig)
        
    
    # Plot Gondensate Rate
    def plot_con_rate(t, profiles, P90, p50, P10, title, ylabel):
        fig, ax = plt.subplots(figsize=(10, 6))
        for profile in profiles:
            ax.plot(t, profile, color="green", alpha=0.1)
        ax.plot(t, P90, color="red", linestyle="--", linewidth=2, label="P90")
        ax.plot(t, p50, color="green", linestyle="--", linewidth=2, label="P50")
        ax.plot(t, P10, color="purple", linestyle="--", linewidth=2, label="P10")
        ax.set_title(title)
        ax.set_xlabel("Time (years)")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid()
        st.pyplot(fig)    
        plt.close(fig)
        
    # Function to calculate CGR based on normalized cumulative gas and type curve
    def calculate_cgr_data(normalized_cum_gas, type_curve):
        cgr_results = []
        if type_curve is not None:
            for scenario in normalized_cum_gas:
                interpolated_cgr = np.interp(
                    scenario,
                    type_curve["Normalized Cumulative Gas (Bcf)"],
                    type_curve["CGR (bbl/MMscf)"],
                    left=0,
                    right=np.max(type_curve["CGR (bbl/MMscf)"])
                )
                cgr_results.append(interpolated_cgr)
        return np.array(cgr_results)
    
    # Function to calculate WGR based on normalized cumulative gas and type curve
    def calculate_wgr_data(normalized_cum_gas, type_curve):
        wgr_results = []
        if type_curve is not None:
            for scenario in normalized_cum_gas:
                interpolated_wgr = np.interp(
                    scenario,
                    type_curve["Normalized Cumulative Gas (Bcf)"],
                    type_curve["WGR (bbl/MMscf)"],
                    left=0,
                    right=np.max(type_curve["WGR (bbl/MMscf)"])
                )
                wgr_results.append(interpolated_wgr)
        return np.array(wgr_results)
    
    # Function to calculate Total Water Rate
    def calculate_water_rate(production_profiles, water_cut_profiles):
        total_water_rate = []
        for oil_rate, water_cut in zip(production_profiles, water_cut_profiles):
            # Cap water cut at 0.99 to avoid division by zero
            water_cut = np.clip(water_cut, 0, 0.99)
            water_rate = oil_rate * (water_cut / (1 - water_cut))
            total_water_rate.append(water_rate)
        return np.array(total_water_rate)
    
    # Function to calculate Total Gas Rate based on GOR
    def calculate_gas_rate(production_profiles, gor_results):
        return production_profiles * gor_results / 1000
    
    # # Function to plot type curve results
    def plot_type_curve_results(results, P90, p50, P10, title, ylabel):
        fig, ax = plt.subplots(figsize=(10, 6))
        for profile in results:
            ax.plot(t, profile, color=profile_color, alpha=0.1)  # Adjust the color as needed            
        # Plot P90, P50, and P10 percentiles
        ax.plot(t, P90, color="red", linestyle="--", linewidth=2, label="P90")
        ax.plot(t, p50, color="green", linestyle="--", linewidth=2, label="P50")
        ax.plot(t, P10, color="purple", linestyle="--", linewidth=2, label="P10")        
        # Adding title and labels
        ax.set_title(title)
        ax.set_xlabel("Time (years)")
        ax.set_ylabel(ylabel)
        
        # Display the legend
        ax.legend()
        ax.grid(True)
        
        # Display the plot
        st.pyplot(fig)
        plt.close(fig)
    
    # Gas Field Specific Options
    if field_type == "Gas Field":
        # st.sidebar.subheader("Gas Field Calculations")
    
        # # File uploader for Gas Type Curve CSV
        # uploaded_gas_type_curve = st.sidebar.file_uploader(
        #     "Upload Gas Type Curve CSV",
        #     type=["csv"],
        #     help="Ensure the CSV contains columns: 'Normalized Cumulative Gas (Bcf)', 'CGR (bbl/MMscf)', 'WGR (bbl/MMscf)'."
        # )
    
        # Load and parse gas type curve
        gas_type_curve = load_gas_type_curve(uploaded_gas_type_curve)
    
        # Proceed only if type curve is uploaded
        if gas_type_curve is not None:
            # Normalize cumulative gas
            normalized_cum_gas = normalize_cumulative_gas(cumulative_productions)
    
            # Calculate CGR and WGR if selected
            if calculate_cgr or calculate_wgr:
                if calculate_cgr:
                    st.header("Condensate Gas Ratio Plot")
                    cgr_results = calculate_cgr_data(normalized_cum_gas, gas_type_curve)
                    P90_cgr, P50_cgr, P10_cgr = np.percentile(cgr_results, [10, 50, 90], axis=0)
                    plot_gas_rate(t, cgr_results, P90_cgr, P50_cgr, P10_cgr, "CGR over Time", "CGR (bbl/MMscf)")
    
                if calculate_wgr:
                    st.header("Water Gas Ratio Plot")
                    wgr_results = calculate_wgr_data(normalized_cum_gas, gas_type_curve)
                    P90_wgr, P50_wgr, P10_wgr = np.percentile(wgr_results, [10, 50, 90], axis=0)
                    plot_water_rate(t, wgr_results, P90_wgr, P50_wgr, P10_wgr, "WGR over Time", "WGR (bbl/MMscf)")
    
            # Calculate and plot Condensate and Water Rates
            if calculate_cgr:
                st.header("Condensate Rate Plot")
                condensate_rate_profiles = np.array(production_profiles) * cgr_results / 1000  # Convert CGR to Mstb/d
                P90_condensate, P50_condensate, P10_condensate = np.percentile(condensate_rate_profiles, [10, 50, 90], axis=0)
                plot_con_rate(t, condensate_rate_profiles, P90_condensate, P50_condensate, P10_condensate,"Condensate Rate over Time", "Condensate Rate (Mstb/d)")
    
            if calculate_wgr:
                st.header("Water Rate Plot")
                water_rate_profiles = np.array(production_profiles) * wgr_results / 1000  # Convert WGR to Mstb/d
                P90_water, P50_water, P10_water = np.percentile(water_rate_profiles, [10, 50, 90], axis=0)
                plot_water_rate(
                    t, water_rate_profiles, P90_water, P50_water, P10_water,
                    "Water Rate over Time", "Water Rate (Mstb/d)"
                )


    #------------------------------------------------------------------------------------------------------------------------------------
    # Scenario descriptions
    #------------------------------------------------------------------------------------------------------------------------------------
    def describe_scenario(scenario, well_schedule, well_productions, field_type):
        def get_single_value(value):
            if value is None:
                return 0.0
            if isinstance(value, (tuple, list, np.ndarray)):
                if len(value) == 0:
                    return 0.0
                return float(np.mean(value))
            return float(value)

        is_gas = field_type == "Gas Field"
        rate_name = "Initial Gas Rate (MMscf/d)" if is_gas else "Initial Oil Rate (Mstb/d)"
        cumulative_name = "Cumulative Gas (Bcf)" if is_gas else "Cumulative Oil (MMSTB)"

        producers = [w for w in well_schedule if str(w.get("well", "")).startswith("P")]
        water_injectors = [w for w in well_schedule if str(w.get("well", "")).startswith("WI")]
        gas_injectors = [w for w in well_schedule if str(w.get("well", "")).startswith("GI")]

        description = f"Field Type: {field_type}\n"
        description += f"Number of Producers: {len(producers)}\n"
        if water_injectors:
            description += f"Number of Water Injectors: {len(water_injectors)}\n"
        if gas_injectors:
            description += f"Number of Gas Injectors: {len(gas_injectors)}\n"

        description += "Scenario Details:\n"
        description += f"{rate_name}: {get_single_value(scenario.get('initial_rate', 0)):.2f}\n"
        description += f"Decline Rate: {get_single_value(scenario.get('decline_rate', 0)):.2f}%\n"
        description += f"Decline Parameter (b): {get_single_value(scenario.get('b', 0)):.2f}\n"
        description += f"Incremental Factor: {get_single_value(scenario.get('incremental_factor', 1.0)):.2f}\n"

        if scenario.get("drilling_duration") is not None:
            description += f"Drilling Duration: {get_single_value(scenario.get('drilling_duration')):.0f} days\n"

        description += "\nProducer Details:\n"
        for producer_idx, well in enumerate(producers):
            start_date = project_start_date + timedelta(days=float(well["start"]) * 365.25)
            cumulative_production = 0.0
            if producer_idx < len(well_productions):
                cumulative_production = (
                    np.sum(well_productions[producer_idx] * 30.4375) / 1000
                )
            description += (
                f"  {well['well']}: Rate={well.get('rate', 0):.2f}, "
                f"Start Date={start_date.strftime('%Y-%m-%d')}, "
                f"{cumulative_name}={cumulative_production:.2f}\n"
            )

        return description

    st.subheader("Scenario Descriptions")

    st.text("P90 Scenario Description")
    st.text(describe_scenario(
        scenarios[P90_index],
        well_schedules[P90_index],
        all_well_productions[P90_index],
        field_type,
    ))

    st.text("P50 Scenario Description")
    st.text(describe_scenario(
        scenarios[P50_index],
        well_schedules[P50_index],
        all_well_productions[P50_index],
        field_type,
    ))

    st.text("P10 Scenario Description")
    st.text(describe_scenario(
        scenarios[P10_index],
        well_schedules[P10_index],
        all_well_productions[P10_index],
        field_type,
    ))

    # EXPORTING SECTION
    # -------------------------------------------------------------------------------------------------------
    total_steps = len(scenarios) + (2 if export_csvs else 0)  # Add 2 steps if CSV export is enabled
    
    # Function to convert DataFrame to CSV and generate a download button
    def generate_csv_download_button(dataframe, file_name, button_label):
        """
        Generate a CSV download button for a given DataFrame.
        """
        csv_buffer = io.StringIO()
        dataframe.to_csv(csv_buffer, index=False)
        st.download_button(
            label=button_label,
            data=csv_buffer.getvalue(),
            file_name=file_name,
            mime="text/csv",
        )
    
    # Step: Prepare CSVs (only if export_csvs is checked)
    if export_csvs:
        st.text("Preparing CSV files...")
    
        # Create `well_results.csv`
        dates = [project_start_date + timedelta(days=365.25 * x) for x in t]
        well_results = []
    
        for scenario_idx, (scenario, well_schedule, well_productions) in enumerate(zip(scenarios, well_schedules, all_well_productions)):
            for well_idx, (well_data, production) in enumerate(zip(well_schedule, well_productions)):
                cumulative_production = np.cumsum(production * 30.4375) / 1000
                incremental_production = cumulative_production * scenario["incremental_factor"]
                for time_idx, date in enumerate(dates):
                    if production[time_idx] > 0:  # Only include times where the well is producing
                        well_results.append({
                            "Date": date,
                            "Scenario": scenario_idx + 1,
                            "Well": well_data["well"],
                            rate_label: production[time_idx],  # Dynamic Rate Label
                            cum_label: cumulative_production[time_idx],  # Dynamic Cumulative Label
                            incr_label: incremental_production[time_idx]  # Dynamic Incremental Label
                        })
    
        well_results_df = pd.DataFrame(well_results)
    
        # Update progress bar and text after well results are ready
        current_progress += 1
        percentage = int(current_progress / total_steps * 100)
        progress_bar.progress(percentage)
        progress_text.text(f"Progress: {percentage}%")
    
        # Create `total_results.csv`
        total_results = []
    
        for scenario_idx, (production_profile, cumulative_production, incremental_production) in enumerate(zip(production_profiles, cumulative_productions, incremental_productions)):
            for time_idx, date in enumerate(dates):
                total_results.append({
                    "Date": date,
                    "Scenario": scenario_idx + 1,
                    rate_label: production_profile[time_idx],  # Dynamic Rate Label
                    cum_label: cumulative_production[time_idx],  # Dynamic Cumulative Label
                    incr_label: incremental_production[time_idx]  # Dynamic Incremental Label
                })
    
        total_results_df = pd.DataFrame(total_results)
    
        # Update progress bar and text after total results are ready
        current_progress += 1
        progress_bar.progress(100)  # Mark as complete
        progress_text.text("Progress: 100% (Complete)")
    
        # Display download buttons
        st.header("Export Results")
    
        # Button to export Well-by-Well Results
        st.subheader("Export Well-by-Well Results")
        generate_csv_download_button(well_results_df, "well_results.csv", "Download Well-by-Well Results")
    
        # Button to export Total Rates
        st.subheader("Export Total Rates")
        generate_csv_download_button(total_results_df, "total_results.csv", "Download Total Rates")
    else:
        # Complete progress bar without CSV generation
        progress_bar.progress(100)
        progress_text.text("Progress: 100% (Complete)")
    
        st.text("CSV export skipped. Use the checkbox to enable CSV generation.")

    #------------------------------------------------------------------------------------------------------------------------------------
        print("Simulation Completed")

    #     st.success("Simulation completed successfully!")
    # else:
    #     st.write("Click 'Run Simulation' in the sidebar to start the Hyperion Simulation.")

    # Project saving is handled in the sidebar Simulation Control section above.

else:
    st.warning("Please create or load a project to proceed.")












