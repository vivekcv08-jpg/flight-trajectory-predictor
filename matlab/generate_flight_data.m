% matlab/generate_flight_data.m
% Generates 5,000 flight trajectories using ODE45 with event detection
% Exports: data/flight_dataset.csv
%
% Physics: 2D point-mass with quadratic aerodynamic drag
%   dx/dt  = vx
%   dy/dt  = vy
%   dvx/dt = -(k_drag/m) * v * vx
%   dvy/dt = -g - (k_drag/m) * v * vy

clear; clc;

%% Constants
g      = 9.81;
rho0   = 1.225;
Cd     = 0.30;
A      = 0.02;
k_drag = 0.5 * rho0 * Cd * A;   % = 0.003675 kg/m

%% Parameter grid
velocities = linspace(20, 300, 20);              % 20 values
angles_deg = linspace(5, 85, 25);               % 25 values
masses     = logspace(log10(0.5), log10(50), 10); % 10 log-spaced

%% Pre-allocate output
N   = numel(velocities) * numel(angles_deg) * numel(masses);
out = zeros(N, 8);   % [v, angle_deg, mass, range, max_alt, flight_time, apex_x, impact_speed]
idx = 0;

%% ODE options
opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-8, ...
              'Events', @trajectory_events, 'MaxStep', 0.5);

tspan = [0, 1000];

for vi = 1:numel(velocities)
    v0 = velocities(vi);
    for ai = 1:numel(angles_deg)
        theta = angles_deg(ai) * pi / 180;
        vx0   = v0 * cos(theta);
        vy0   = v0 * sin(theta);
        for mi = 1:numel(masses)
            m  = masses(mi);
            y0 = [0; 0; vx0; vy0];

            % ODE function (capture m and k_drag via closure)
            ode_fun = @(t, y) eom(t, y, g, k_drag, m);

            % Integrate to apex first (vy = 0 event)
            opts_apex = odeset(opts, 'Events', @apex_event);
            [~, Y_asc, ~, YE_apex, ~] = ode45(ode_fun, tspan, y0, opts_apex);

            if ~isempty(YE_apex)
                apex_x   = YE_apex(end, 1);
                max_alt  = YE_apex(end, 2);
                y0_desc  = YE_apex(end, :)';
            else
                apex_x  = Y_asc(end, 1);
                max_alt = Y_asc(end, 2);
                y0_desc = Y_asc(end, :)';
            end

            % Integrate from apex to ground impact (y = 0 event)
            opts_land = odeset(opts, 'Events', @ground_event);
            [~, ~, ~, YE_land, ~] = ode45(ode_fun, tspan, y0_desc, opts_land);

            if ~isempty(YE_land)
                land_x       = YE_land(end, 1);
                land_vy      = YE_land(end, 4);
                land_vx      = YE_land(end, 3);
                impact_speed = hypot(land_vx, land_vy);
            else
                % Fallback: use end of integration
                land_x       = Y_asc(end, 1);
                impact_speed = hypot(Y_asc(end, 3), Y_asc(end, 4));
            end

            % Full flight time via total integration
            opts_full = odeset(opts, 'Events', @ground_event);
            [T_full, ~, ~, ~, ~] = ode45(ode_fun, tspan, y0, opts_full);
            flight_time = T_full(end);

            idx = idx + 1;
            out(idx, :) = [v0, angles_deg(ai), m, land_x, max_alt, flight_time, apex_x, impact_speed];
        end
    end
    fprintf('Velocity %d/%d complete\n', vi, numel(velocities));
end

%% Export CSV
header = {'velocity','angle','mass','range','max_altitude','flight_time','apex_x','impact_speed'};
T = array2table(out(1:idx,:), 'VariableNames', header);
writetable(T, '../data/flight_dataset.csv');
fprintf('Saved %d rows to data/flight_dataset.csv\n', idx);


%% Local event functions
function dydt = eom(~, y, g, k_drag, m)
    vx = y(3); vy = y(4);
    v  = hypot(vx, vy);
    k  = k_drag / m;
    dydt = [vx; vy; -k*v*vx; -g - k*v*vy];
end

function [value, isterminal, direction] = apex_event(~, y)
    % Trigger when vy = 0 (peak altitude), on descent
    value      = y(4);     % vy
    isterminal = 1;
    direction  = -1;       % only trigger when vy decreasing through 0
end

function [value, isterminal, direction] = ground_event(~, y)
    % Trigger when y = 0 on descent
    value      = y(2);     % altitude
    isterminal = 1;
    direction  = -1;       % only trigger when y decreasing through 0
end

function [value, isterminal, direction] = trajectory_events(~, y)
    value      = [y(2); y(4)];
    isterminal = [1; 0];
    direction  = [-1; -1];
end
