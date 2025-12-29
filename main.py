from math import inf
import numpy as np
import dash
import scipy.interpolate
from dash import html, dcc
from dash import Input, Output, State
from numpy import cos, radians, sin, clip
import plotly.graph_objects as go


VEHICLE_MODELS = {
    "Renault Laguna 3 2.0 dCi (150 KM)": {
        "mass": 1450,
        "area": 2.20,
        "Cd": 0.30,
        "roll": 0.015,
        "wheel_radius": 0.32,
        "fd_ratio": 3.4,
        "gear_ratios": "3.8, 2.0, 1.4, 1.0, 0.8, 0.65",
        "rpm_points": "800, 1500, 2000, 3000, 3800, 4500",
        "torque_points": "180, 300, 340, 320, 280, 200",
        "shift_up": 3800,
        "shift_down": 1600
    },
    "Toyota Aygo 1.0 (69 KM)": {
        "mass": 950,
        "area": 2.05,
        "Cd": 0.32,
        "roll": 0.015,
        "wheel_radius": 0.28,
        "fd_ratio": 3.7,
        "gear_ratios": "3.5, 1.9, 1.3, 0.95, 0.75",
        "rpm_points": "1000, 2500, 4000, 5000, 6000, 6500",
        "torque_points": "70, 85, 95, 90, 80, 60",
        "shift_up": 5500,
        "shift_down": 2000
    },
    "Ford Mustang GT 5.0 V8 (450 KM)": {
        "mass": 1850,
        "area": 2.25,
        "Cd": 0.32,
        "roll": 0.012,
        "wheel_radius": 0.35,
        "fd_ratio": 3.55,
        "gear_ratios": "4.7, 3.0, 2.2, 1.5, 1.0, 0.75",
        "rpm_points": "1000, 2500, 4600, 6000, 7000, 7500",
        "torque_points": "350, 450, 529, 500, 400, 300",
        "shift_up": 6800,
        "shift_down": 2500
    },

    "Renault T 480 bez naczepy (Solo)": {
        "mass": 7500,
        "area": 10.0,
        "Cd": 0.55,
        "roll": 0.006,
        "wheel_radius": 0.52,
        "fd_ratio": 2.5,
        "gear_ratios": "15.0, 10.0, 7.0, 5.0, 3.5, 2.5, 1.8, 1.3, 1.0",
        "rpm_points": "700, 1000, 1200, 1400, 1600, 2000",
        "torque_points": "1000, 2200, 2400, 2400, 2000, 1500",
        "shift_up": 1800,
        "shift_down": 1000
    },
    "Renault T 480 z ładunkiem 20 ton": {
        "mass": 28500,
        "area": 10.0,
        "Cd": 0.75,
        "roll": 0.006,
        "wheel_radius": 0.52,
        "fd_ratio": 2.5,
        "gear_ratios": "15.0, 10.0, 7.0, 5.0, 3.5, 2.5, 1.8, 1.3, 1.0",
        "rpm_points": "700, 1000, 1200, 1400, 1600, 2000",
        "torque_points": "1000, 2200, 2400, 2400, 2000, 1500",
        "shift_up": 1800,
        "shift_down": 1000
    },
    "Volvo FH bez naczepy (Solo)": {
        "mass": 8500,
        "area": 10.0,
        "Cd": 0.55,
        "roll": 0.006,
        "wheel_radius": 0.52,
        "fd_ratio": 2.5,
        "gear_ratios": "15.0, 10.0, 7.0, 5.0, 3.5, 2.5, 1.8, 1.3, 1.0",
        "rpm_points": "700, 1000, 1200, 1400, 1600, 2000",
        "torque_points": "1000, 2200, 2400, 2400, 2000, 1500",
        "shift_up": 1800,
        "shift_down": 1000
    },
    "Volvo FH z ładunkiem 30 ton": {
        "mass": 40000,
        "area": 10.0,
        "Cd": 0.75,
        "roll": 0.006,
        "wheel_radius": 0.52,
        "fd_ratio": 2.5,
        "gear_ratios": "15.0, 10.0, 7.0, 5.0, 3.5, 2.5, 1.8, 1.3, 1.0",
        "rpm_points": "700, 1000, 1200, 1400, 1600, 2000",
        "torque_points": "1000, 2200, 2400, 2400, 2000, 1500",
        "shift_up": 1800,
        "shift_down": 1000
    },
    "Bugatti Chiron SS 300+ (1600 KM)": {
        "mass": 2000,
        "area": 2.50,
        "Cd": 0.38,
        "roll": 0.010,
        "wheel_radius": 0.38,
        "fd_ratio": 2.55,
        "gear_ratios": "5.5, 3.5, 2.5, 1.8, 1.3, 1.0, 0.8, 0.6",
        "rpm_points": "1000, 2500, 4000, 5500, 6500, 7000",
        "torque_points": "800, 1200, 1500, 1600, 1550, 1400",
        "shift_up": 6700,
        "shift_down": 2000
    },
}



def torque_at_rpm(rpm):
    rpm = clip(rpm, rpm_points[0], rpm_points[-1])
    return np.interp(rpm, rpm_points, torque_points)


def calculate_rpm(vehicle_speed, gear_index):
    if wheel_radius == 0:
        return 0.0

    wheel_speed_rad_s = vehicle_speed / wheel_radius
    wheel_rpm = wheel_speed_rad_s / (2 * np.pi) * 60


    engine_rpm = wheel_rpm * gear_ratios[gear_index] * final_drive_ratio
    return engine_rpm

def engine_force(vehicle_speed, gear, throttle):

    wheel_rpm = vehicle_speed / (2 * np.pi * wheel_radius) * 60
    engine_rpm = wheel_rpm * gear_ratios[gear] * final_drive_ratio


    torque = torque_at_rpm(engine_rpm) * throttle

    if throttle > 0:
        torque *= throttle
    else:
        torque *= -0.3


    return torque * gear_ratios[gear] * final_drive_ratio / wheel_radius, engine_rpm


def shift_gear(engine_rpm, gear):
    global shift_up_rpm, shift_down_rpm

    if engine_rpm > shift_up_rpm and gear < len(gear_ratios) - 1:
        return gear + 1
    if engine_rpm < shift_down_rpm and gear > 0:
        return gear - 1
    return gear


# =============================================================


def calculate_drag(
    vehicle_speed: float,
    rolling_resistance_coefficient: float,
    slope: float,
    air_density: float,
    front_area: float,
    vehicle_mass: float,
    drag_coefficient: float,
    gravity: float,
) -> float:
    slope_radians = radians(slope)

    rolling_resistances = (
        rolling_resistance_coefficient
        * vehicle_mass
        * gravity
        * cos(slope_radians)
    )

    air_resistances = (
        0.5 * air_density * front_area * drag_coefficient * vehicle_speed ** 2
    )

    slope_resistance = vehicle_mass * gravity * sin(slope_radians)

    return rolling_resistances + air_resistances + slope_resistance


def calculate_figure():
    global simulation_time, start_velocity, target_velocity
    global rolling_resistance_coefficient, slope, air_density
    global front_area, vehicle_mass, drag_coefficient
    global gravity, accuracy

    # zaczyna od poczatkowej predkosci
    current_velocity = start_velocity
    current_gear = 0

    # poczatkowy blad
    current_error = target_velocity - start_velocity

    error_accumulator = 0.0
    previous_error = current_error

    simulation_steps = int(simulation_time / accuracy)

    velocity_history = [current_velocity]
    time_history = [0.0]
    rpm_history = [0.0]

    gear_shifts = []

    best_start_gear = 0


    if current_velocity > 0:
        for gear_index in range(len(gear_ratios) - 1, -1, -1):

            rpm = calculate_rpm(current_velocity, gear_index)

            if rpm >= shift_down_rpm or gear_index == 0:
                best_start_gear = gear_index
                break

    current_gear = best_start_gear

    for i in range(1, simulation_steps + 1):

        resistances = calculate_drag(
            current_velocity,
            rolling_resistance_coefficient,
            slope,
            air_density,
            front_area,
            vehicle_mass,
            drag_coefficient,
            gravity,
        )

        current_error = target_velocity - current_velocity

        
        throttle = calculate_signal(
            current_error,
            previous_error,
            error_accumulator,
            accuracy,
        )
        throttle = clip(throttle, 0.0, 1.0)

        
        engine_f, engine_rpm = engine_force(
            current_velocity,
            current_gear,
            throttle,
        )

        
        old_gear = current_gear
        current_gear = shift_gear(engine_rpm, current_gear)
        if(old_gear != current_gear):
            gear_shifts.append((
                i * accuracy,
                current_velocity * 3.6,  # km/h
                current_gear
            ))

        
        acceleration = (engine_f - resistances) / vehicle_mass
        #acceleration = clip(acceleration, -inf, max_acceleration)

        current_velocity += acceleration * accuracy
        current_velocity = clip(current_velocity, 0, inf)

        previous_error = current_error

        if throttle < 1.0:
            error_accumulator += current_error * accuracy

        velocity_history.append(current_velocity)
        time_history.append(i * accuracy)
        rpm_history.append(engine_rpm)

    velocity_history = [v * 3.6 for v in velocity_history]

    return time_history, velocity_history, gear_shifts, rpm_history


def calculate_signal(
    error: float,
    previous_error: float,
    accumulated_error: float,
    probing: float,
) -> float:
    global K_p, K_i, K_d

    P = K_p * error
    I = K_i * accumulated_error
    D = K_d / probing * (error - previous_error)

    return P + I + D




# wartosci sterujące
max_acceleration = 2.5  # m/s^2
vehicle_mass = 1450  # kg
front_area = 2.1  # m^2
drag_coefficient = 0.29
rolling_resistance_coefficient = 0.015
slope = 0.0  # degrees
air_density = 1.204

start_velocity = 0.0 / 3.6
target_velocity = 120 / 3.6
min_velocity = 40 / 3.6
simulation_time = 60

gravity = 9.81
accuracy = 0.01

K_p = 0.65
K_i = 0.0
K_d = 0.6

shift_up_rpm = 2500
shift_down_rpm = 1300

gear_ratios = [3.6, 2.1, 1.4, 1.0, 0.8]

final_drive_ratio = 3.9

wheel_radius = 0.3

rpm_points = np.array([
    800,
    1200,
    1800,
    2500,
    3200,
    4000,
    4800,
    5500,
    6200
])

torque_points = np.array([
     70,
    110,
    160,
    200,
    230,
    260,
    255,
    235,
    190
])


app = dash.Dash(__name__)

# Add a button under the sliders
app.layout = html.Div([
    html.H1("Symulacja tempomatu"),
    # Store for keeping the previous simulation curve (time and speed)
    dcc.Store(id="prev_curve"),
    dcc.Graph(id="velocity_graph"),


    dcc.Button("Uruchom symulację", id="recalc_button", n_clicks=0, style={'marginTop': '20px'}),
    dcc.Dropdown(
        id='model_selector',
        options=[
            {'label': name, 'value': name}
            for name in VEHICLE_MODELS.keys()
        ],
        placeholder="Wybierz predefiniowany model pojazdu...",
        clearable=False,
        style={'marginBottom': '30px', 'width': '70%'}
    ),
    html.Hr(),
    html.Div([
        html.Div([
            html.Label("Masa pojazdu (kg)"),
            dcc.Slider(id='vehicle_mass', min=10, max=50000, step=10, value=1450,
                       marks={10: '10', 25000: '25k', 50000: '50k'}),
            html.Label("Powierzchnia czołowa pojazdu (m²)"),
            dcc.Slider(id='front_area', min=0, max=10, step=0.1, value=2.1,
                       marks={0: '0', 5: '5', 10: '10'}),
            html.Label("Współczynnik oporu aerodynamicznego"),
            dcc.Slider(id='drag_coefficient', min=0, max=1, step=0.01, value=0.29,
                       marks={0: '0', 0.5: '0.5', 1: '1'}),
            html.Label("Współczynnik oporu toczenia"),
            dcc.Slider(id='rolling_resistance', min=0, max=0.05, step=0.001, value=0.015,
                       marks={0: '0', 0.025: '0.025', 0.05: '0.05'}),
        ], style={'width': '48%', 'marginBottom': '20px'}),

        html.Div([

            html.Label("Prędkość początkowa (km/h)"),
            dcc.Slider(id='initial_velocity', min=0, max=200, step=1, value=0,
                       marks={0: '0', 100: '100', 200: '200'}),
            html.Label("Prędkość zadana (km/h)"),
            dcc.Slider(id='target_velocity', min=0, max=600, step=1, value=120,
                       marks={0: '0', 100: '100', 200: '200',300: '300', 400: '400'}),
            html.Label("Czas symulacji (s)"),
            dcc.Slider(id='simulation_time', min=10, max=1000, step=10, value=60,
                       marks={10: '10', 500: '500', 1000: '1000'}),
            html.Hr(),
            html.H4("Parametry PID"),
            html.Label("człon P"),
            dcc.Slider(id='Kp', min=0, max=10, step=0.01, value=0.65,
                       marks={0: '0', 5: '5', 10: '10'}),
            html.Label("człon I"),
            dcc.Slider(id='Ki', min=0, max=1, step=0.001, value=0.0,
                       marks={0: '0', 0.5: '0.5', 1: '1'}),
            html.Label("człon D"),
            dcc.Slider(id='Kd', min=0, max=10, step=0.01, value=0.6,
                       marks={0: '0', 5: '5', 10: '10'}),

        ], style={'width': '48%', 'marginBottom': '20px'}),
    ], style={
        'display': 'flex',
        'justifyContent': 'space-between',
        'flexWrap': 'wrap'
    }),
    html.Hr(),
    html.H2("Parametry napędu"),
    html.Div([

        # === Lewa Kolumna: Podstawowe Przełożenia ===
        html.Div([
            html.H4("Geometria i Przełożenia"),

            html.Label("Średnica koła (m)"),
            dcc.Input(
                id='wheel_radius',
                type='number',
                value=0.3,
                step=0.01,
                style={'width': '100%', 'marginBottom': '15px'}
            ),

            html.Label("Ostateczne przeniesienie napędu"),
            dcc.Input(
                id='final_drive_ratio',
                type='number',
                value=3.9,
                step=0.01,
                style={'width': '100%', 'marginBottom': '15px'}
            ),

            html.Label("Przełożenia biegów (rozdzielone przecinkami, np. 3.6, 2.1, 1.4)"),
            dcc.Input(
                id='gear_ratios_input',
                type='text',
                value='3.6, 2.1, 1.4, 1.0, 0.8',
                style={'width': '100%', 'marginBottom': '30px'}
            ),

            html.H4("Zmiany Biegów (Automat typu DSG)"),
            html.Label("Obroty zmiany biegu w górę"),
            dcc.Slider(id='shift_up_rpm', min=1500, max=7000, step=100, value=2500,
                       marks={1500: '1.5k', 4000: '4k', 7000: '7k'}),

            html.Label("Obroty zmiany biegu w dół"),
            dcc.Slider(id='shift_down_rpm', min=500, max=3000, step=50, value=1300,
                       marks={500: '0.5k', 1500: '1.5k', 3000: '3k'}),
            html.H5("Punkty RPM (Obroty - Oś X, rozdzielone przecinkami)"),
            dcc.Input(
                id='rpm_input',
                type='text',
                value='800, 1200, 1800, 2500, 3200, 4000, 4800, 5500, 6200',
                style={'width': '100%', 'marginBottom': '10px'}
            ),

            html.H5("Punkty Momentu (Nm - Oś Y, rozdzielone przecinkami)"),
            dcc.Input(
                id='torque_input',
                type='text',
                value='70, 110, 160, 200, 230, 260, 255, 235, 190',
                style={'width': '100%'}
            ),
        ], style={'width': '48%', 'paddingRight': '2%'}),

        # === Prawa Kolumna: Krzywa Momentu ===
        html.Div([
            html.H4("Krzywa Momentu Obrotowego"),

            # Wykres wyświetlający krzywą na żywo (wymaga callbacku)
            dcc.Graph(id="torque_curve_graph", style={'marginBottom': '15px'}),
            dcc.Button("Zapisz parametry napędu", id="torque_calc_button", n_clicks=0, style={'marginBottom': '15px'}),

        ], style={'width': '48%', 'paddingLeft': '2%', 'verticalAlign': 'top'})

    ], style={
        'display': 'flex',
        'justifyContent': 'space-between',
        'flexWrap': 'wrap'
    })

])




@app.callback(
    # OUTPUTS: Zmiana wartości w polach
    Output('vehicle_mass', 'value'),
    Output('front_area', 'value'),
    Output('drag_coefficient', 'value'),
    Output('rolling_resistance', 'value'),
    Output('final_drive_ratio', 'value'),
    Output('gear_ratios_input', 'value'),
    Output('rpm_input', 'value'),
    Output('torque_input', 'value'),
    Output('shift_up_rpm', 'value'),
    Output('shift_down_rpm', 'value'),
    Output("wheel_radius", "value"),
    Output("torque_curve_graph", "figure", allow_duplicate=True),


    # INPUT: Wybór modelu
    Input('model_selector', 'value'),
    prevent_initial_call=True,
    # State: Opcjonalnie, aby uniknąć uruchamiania przy ładowaniu
)
def update_parameters_from_model(selected_model_name):
    # Warunek początkowy: jeśli nic nie wybrano, nie zmieniaj nic
    if not selected_model_name:
        # Zwracamy tuple pustych wartości lub None dla wszystkich Output
        # None oznacza brak aktualizacji (choć Dash może wymagać konkretnych wartości)
        return (dash.no_update,) * 11

    model_data = VEHICLE_MODELS[selected_model_name]

    fig_curve = go.Figure()
    rpm_array = np.array([])
    torque_array = np.array([])
    gear_ratios_array = np.array([])

    try:

        rpm_array = np.array([float(x) for x in model_data["rpm_points"].split(",")])
        torque_array = np.array([float(x) for x in model_data["torque_points"].split(",")])

        gear_ratios_array = np.array([float(x) for x in model_data["gear_ratios"].split(",")])

        if len(rpm_array) != len(torque_array):
            raise ValueError("Liczba punktów RPM i Momentu musi być równa i wynosić co najmniej 2.")



        interpolated_torque_f = scipy.interpolate.interp1d(rpm_array, torque_array, kind="cubic")
        new_rpm_points = np.linspace(rpm_array.min(), rpm_array.max(), 1000)
        new_torque_points = interpolated_torque_f(new_rpm_points)

        fig_curve.add_scatter(x=new_rpm_points, y=new_torque_points, mode="lines", line=dict(width=3, color="black"))
        fig_curve.update_layout(template="plotly_white", xaxis_title = "RPM", yaxis_title = "Moment (Nm)")
    except ValueError as e:
        fig_curve.add_annotation(text=f"Błąd danych: {e}", showarrow=False, font=dict(size=14, color="red"))
        fig_curve.update_layout(title="Krzywa Momentu Silnika (Błąd)", height=300)


    # Zwracamy tuple wartości w tej samej kolejności, w jakiej są Output

    global rpm_points, torque_points, gear_ratios, wheel_radius, final_drive_ratio, shift_up_rpm, shift_down_rpm, vehicle_mass, front_area, drag_coefficient, rolling_resistance

    rpm_points = rpm_array
    torque_points = torque_array

    vehicle_mass = model_data["mass"]
    front_area = model_data["area"]
    drag_coefficient = model_data["Cd"]
    rolling_resistance = model_data["roll"]
    final_drive_ratio = model_data["fd_ratio"]
    gear_ratios = gear_ratios_array
    shift_up_rpm = model_data["shift_up"]
    shift_down_rpm = model_data["shift_down"]

    return (
        model_data["mass"],
        model_data["area"],
        model_data["Cd"],
        model_data["roll"],
        model_data["fd_ratio"],
        model_data["gear_ratios"],
        model_data["rpm_points"],
        model_data["torque_points"],
        model_data["shift_up"],
        model_data["shift_down"],
        model_data["wheel_radius"],
        fig_curve
    )

@app.callback(
    Output("torque_curve_graph", "figure"),
    Input("torque_calc_button", "n_clicks"),
    State("rpm_input", "value"),
    State("torque_input", "value"),
    State("wheel_radius", "value"),
    State("final_drive_ratio", "value"),
    State("gear_ratios_input", "value"),
    State("shift_up_rpm", "value"),
    State("shift_down_rpm", "value"),

)
def update_torque(n_clicks, rpm_str, torque_str, wheel_radius_param, final_drive_ratio_param, gear_ratios_str, shift_up_rpm_param, shift_down_rpm_param):
    global rpm_points, torque_points
    global gear_ratios
    global wheel_radius
    global final_drive_ratio
    global shift_up_rpm, shift_down_rpm





    fig_curve = go.Figure()

    try:

        rpm_array = np.array([float(x) for x in rpm_str.split(",")])
        torque_array = np.array([float(x) for x in torque_str.split(",")])

        gear_ratios_array = np.array([float(x) for x in gear_ratios_str.split(",")])

        if len(rpm_array) != len(torque_array):
            raise ValueError("Liczba punktów RPM i Momentu musi być równa i wynosić co najmniej 2.")

        rpm_points = rpm_array
        torque_points = torque_array
        gear_ratios = gear_ratios_array
        wheel_radius = wheel_radius_param
        final_drive_ratio = final_drive_ratio_param
        shift_up_rpm = shift_up_rpm_param
        shift_down_rpm = shift_down_rpm_param

        interpolated_torque_f = scipy.interpolate.interp1d(rpm_array, torque_array, kind="cubic")
        new_rpm_points = np.linspace(rpm_array.min(), rpm_array.max(), 1000)
        new_torque_points = interpolated_torque_f(new_rpm_points)

        fig_curve.add_scatter(x=new_rpm_points, y=new_torque_points, mode="lines", line=dict(width=3, color="black"))
        fig_curve.update_layout(template="plotly_white", xaxis_title = "RPM", yaxis_title = "Moment (Nm)")
    except ValueError as e:
        fig_curve.add_annotation(text=f"Błąd danych: {e}", showarrow=False, font=dict(size=14, color="red"))
        fig_curve.update_layout(title="Krzywa Momentu Silnika (Błąd)", height=300)
    return fig_curve



# Callback to update the figure
@app.callback(
    Output("velocity_graph", "figure"),
    Output("prev_curve", "data"),
    Input("recalc_button", "n_clicks"),
    State("vehicle_mass", "value"),
    State("front_area", "value"),
    State("drag_coefficient", "value"),
    State("rolling_resistance", "value"),
    State("initial_velocity", "value"),
    State("target_velocity", "value"),
    State("simulation_time", "value"),
    State("Kp", "value"),
    State("Ki", "value"),
    State("Kd", "value"),
    State("prev_curve", "data"),
)
def update_graph(n_clicks, mass, area, Cd, roll,
                 v0, v_target, sim_time, Kp_val, Ki_val, Kd_val, prev_curve_data):

    # Set global parameters
    global vehicle_mass, front_area, drag_coefficient
    global rolling_resistance_coefficient, slope, air_density
    global start_velocity, target_velocity, simulation_time
    global K_p, K_i, K_d

    vehicle_mass = mass
    front_area = area
    drag_coefficient = Cd
    rolling_resistance_coefficient = roll
    start_velocity = v0 / 3.6  # km/h -> m/s
    target_velocity = v_target / 3.6
    simulation_time = sim_time
    K_p = Kp_val
    K_i = Ki_val
    K_d = Kd_val

    x, y, shifts, rpms = calculate_figure()

    fig_new = go.Figure()

    # If we have a previous curve stored, show it as a greyed-out line
    if prev_curve_data and isinstance(prev_curve_data, dict):
        prev_x = prev_curve_data.get("x")
        prev_y = prev_curve_data.get("y")
        if prev_x is not None and prev_y is not None:
            fig_new.add_trace(
                go.Scatter(
                    x=prev_x,
                    y=prev_y,
                    mode="lines",
                    line=dict(color="#888888", width=2),
                    opacity=0.4,
                    name="Poprzednia prędkość"
                )
            )

    fig_new.add_trace(go.Scatter(x=x, y=y, mode="lines",
                                   line=dict(color="#007BFF", width=3), name="Prędkość",
                                   customdata=np.stack([rpms],
                                                       axis=-1),
                                   hovertemplate=
                                   "<b>Prędkość:</b> %{y:.1f} km/h<br>" +
                                   "<b>Obroty silnika:</b> %{customdata[0]:.0f} RPM<br>" +  # Użycie customdata[0] dla RPM
                                   "<b>Czas</b>: %{x:.2f} s<br>" +
                                   "<extra></extra>"
                                   ))
    fig_new.add_hline(y=target_velocity*3.6, line_dash="dash", line_color="black")

    oldFig = fig_new
    if shifts:
        shift_times = [s[0] for s in shifts]
        shift_speeds = [s[1] for s in shifts]
        shift_gears = [s[2]+1 for s in shifts]

        fig_new.add_trace(
            go.Scatter(
                x=shift_times,
                y=shift_speeds,
                mode="markers",
                marker=dict(
                    size=10,
                    color=shift_gears,
                    line=dict(width=1, color="black")
                ),
                name="Zmiany biegów",
                hovertemplate=
                "<b>Prędkość:</b> %{y:.1f} km/h<br>"
                "<b>Bieg:</b> %{marker.color}<br>"
                "<b>Czas:</b> %{x:.2f}s<extra></extra>"
            )
        )

    fig_new.update_layout(template="plotly_white", xaxis_title="Czas (s)", yaxis_title="Prędkość (km/h)")

    # Update the stored previous curve with the just-calculated one
    prev_payload = {"x": list(x), "y": list(y)}

    return fig_new, prev_payload


if __name__ == "__main__":
    app.run(host="127.0.0.1",debug=True)
