from math import inf

import dash
from dash import html, dcc
from dash import Input, Output, State
from numpy import cos, radians, sin, clip
import plotly.graph_objects as go




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
    """Oblicza sumę wszystkich oporów działających na pojazd."""
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
    global simulation_time, min_velocity, start_velocity, target_velocity, max_acceleration, rolling_resistance_coefficient, slope, air_density, front_area, vehicle_mass, drag_coefficient, gravity, accuracy
    current_velocity = start_velocity

    # obliczanie uchybu
    current_error = target_velocity - start_velocity
    error_accumulator = 0.0
    previous_error = current_error

    # czas symulacji / czestosc probkowania co sekunde
    simulation_steps = int(simulation_time / accuracy)

    # dane do wykresu
    velocity_history = [current_velocity]
    time_history = [0.0]

    for i in range(1, simulation_steps + 1):

        # obliczanie sily oporów [Niutony]
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

        # obliczanie uchybu
        current_error = target_velocity - current_velocity

        # obliczanie sygnalu sterującego
        pid_signal = calculate_signal(
            current_error,
            previous_error,
            error_accumulator,
            accuracy,
        )

        # "ucinanie" sygnalu sterujacego do zakresu [0 ; maks. przyspieszenie]
        # tempomat nie bedzie "wciskal" hamulca tylko pozwoli oporom(powietrza i toczenia) przyhamowac pojazd
        pid_signal = clip(pid_signal, 0, max_acceleration)

        resistance_acceleration = resistances / vehicle_mass
        acceleration_unlimited = pid_signal - resistance_acceleration
        # hamowanie glownie przez opory
        acceleration_limited = clip(
            acceleration_unlimited,
            -resistance_acceleration,
            max_acceleration,
        )

        current_velocity += acceleration_limited * accuracy
        current_velocity = clip(current_velocity, 0, inf)
        previous_error = current_error

        # anti wind-up
        if acceleration_unlimited == acceleration_limited:
            error_accumulator += current_error * accuracy

        velocity_history.append(current_velocity)
        time_history.append(i * accuracy)

    velocity_history = [v * 3.6 for v in velocity_history]

    return time_history, velocity_history


def calculate_signal(
    error: float,
    previous_error: float,
    accumulated_error: float,
    probing: float,
) -> float:
    """Obliczanie sygnalu sterujacego czyli przyspieszenia/spowolnienia"""
    global K_p, K_i, K_d


    P = K_p * error
    I = K_i * accumulated_error
    D = K_d / probing * (error - previous_error)

    return P + I + D


import matplotlib.pyplot as plt


# wartosci sterujące
max_acceleration = 2.5  # m/s^2
vehicle_mass = 1450  # kg
front_area = 2.1  # m^2
drag_coefficient = 0.29
rolling_resistance_coefficient = 0.015
slope = 0.0  # degrees
air_density = 1.204

start_velocity = 40 / 3.6
target_velocity = 120 / 3.6
min_velocity = 40 / 3.6
simulation_time = 60

gravity = 9.81
accuracy = 0.01

K_p = 0.65
K_i = 0.002
K_d = 2.0

x, y = calculate_figure()


fig = go.Figure(
    go.Scatter(x=x, y=y, mode="lines",
               line=dict(color="#007BFF", width=3)
))
fig.update_layout(template="plotly_white", xaxis_title = "Czas (s)", yaxis_title = "Prędkość (km/h)")


app = dash.Dash(__name__)

# Add a button under the sliders
app.layout = html.Div([
    html.H1("Symulacja tempomatu PID"),
    dcc.Graph(id="velocity_graph", figure=fig),


    dcc.Button("Oblicz", id="recalc_button", n_clicks=0, style={'marginTop': '20px'}),
    html.Hr(),
    html.Div([
        html.Div([
            html.Label("Maks. przyspieszenie (m/s²)"),
            dcc.Slider(id='max_acceleration', min=0, max=10, step=0.1, value=2.5,
                       marks={0: '0', 5: '5', 10: '10'}),
            html.Label("Masa pojazdu (kg)"),
            dcc.Slider(id='vehicle_mass', min=10, max=50000, step=10, value=1450,
                       marks={10: '10', 25000: '25k', 50000: '50k'}),
            html.Label("Powierzchnia czołowa pojazdu (m²)"),
            dcc.Slider(id='front_area', min=0, max=10, step=0.1, value=2.1,
                       marks={0: '0', 5: '5', 10: '10'}),
            html.Label("Współczynnik oporu aero."),
            dcc.Slider(id='drag_coefficient', min=0, max=1, step=0.01, value=0.29,
                       marks={0: '0', 0.5: '0.5', 1: '1'}),
            html.Label("Współczynnik oporu toczenia"),
            dcc.Slider(id='rolling_resistance', min=0, max=0.05, step=0.001, value=0.015,
                       marks={0: '0', 0.025: '0.025', 0.05: '0.05'}),
            html.Label("Nachylenie drogi (°)"),
            dcc.Slider(id='road_slope', min=0, max=90, step=1, value=0,
                       marks={0: '0', 45: '45', 90: '90'}),
            html.Label("Gęstość powietrza (kg/m³)"),
            dcc.Slider(id='air_density', min=0.5, max=2.0, step=0.01, value=1.204,
                       marks={0.5: '0.5', 1.0: '1.0', 1.5: '1.5', 2.0: '2.0'}),
        ], style={'width': '48%', 'marginBottom': '20px'}),

        html.Div([

            html.Label("Prędkość początkowa (km/h)"),
            dcc.Slider(id='initial_velocity', min=0, max=200, step=1, value=40,
                       marks={0: '0', 100: '100', 200: '200'}),
            html.Label("Prędkość zadana (km/h)"),
            dcc.Slider(id='target_velocity', min=0, max=400, step=1, value=120,
                       marks={0: '0', 100: '100', 200: '200',300: '300', 400: 'w chuj'}),
            html.Label("Czas symulacji (s)"),
            dcc.Slider(id='simulation_time', min=10, max=1000, step=10, value=60,
                       marks={10: '10', 500: '500', 1000: '1000'}),
            html.Hr(),
            html.H4("Parametry PID"),
            html.Label("Kp"),
            dcc.Slider(id='Kp', min=0, max=10, step=0.01, value=0.65,
                       marks={0: '0', 5: '5', 10: '10'}),
            html.Label("Ki"),
            dcc.Slider(id='Ki', min=0, max=1, step=0.001, value=0.002,
                       marks={0: '0', 0.5: '0.5', 1: '1'}),
            html.Label("Kd"),
            dcc.Slider(id='Kd', min=0, max=10, step=0.01, value=2.0,
                       marks={0: '0', 5: '5', 10: '10'}),

        ], style={'width': '48%', 'marginBottom': '20px'}),
    ], style={
        'display': 'flex',
        'justifyContent': 'space-between',
        'flexWrap': 'wrap'
    }),

])

# Callback to update the figure
@app.callback(
    Output("velocity_graph", "figure"),
    Input("recalc_button", "n_clicks"),
    State("max_acceleration", "value"),
    State("vehicle_mass", "value"),
    State("front_area", "value"),
    State("drag_coefficient", "value"),
    State("rolling_resistance", "value"),
    State("road_slope", "value"),
    State("air_density", "value"),
    State("initial_velocity", "value"),
    State("target_velocity", "value"),
    State("simulation_time", "value"),
    State("Kp", "value"),
    State("Ki", "value"),
    State("Kd", "value"),
)
def update_graph(n_clicks, max_acc, mass, area, Cd, roll, slope_val, rho,
                 v0, v_target, sim_time, Kp_val, Ki_val, Kd_val):

    # Set global parameters
    global max_acceleration, vehicle_mass, front_area, drag_coefficient
    global rolling_resistance_coefficient, slope, air_density
    global start_velocity, target_velocity, simulation_time
    global K_p, K_i, K_d

    max_acceleration = max_acc
    vehicle_mass = mass
    front_area = area
    drag_coefficient = Cd
    rolling_resistance_coefficient = roll
    slope = slope_val
    air_density = rho
    start_velocity = v0 / 3.6  # km/h -> m/s
    target_velocity = v_target / 3.6
    simulation_time = sim_time
    K_p = Kp_val
    K_i = Ki_val
    K_d = Kd_val

    x, y = calculate_figure()
    fig_new = go.Figure(go.Scatter(x=x, y=y, mode="lines",
                                   line=dict(color="#007BFF", width=3)))
    fig_new.update_layout(template="plotly_white", xaxis_title="Czas (s)", yaxis_title="Prędkość (m/s)")
    return fig_new


if __name__ == "__main__":
    app.run(host="127.0.0.1",debug=True)
