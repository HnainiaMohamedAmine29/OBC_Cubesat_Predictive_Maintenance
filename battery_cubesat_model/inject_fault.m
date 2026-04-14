function [p_out, fault_state] = inject_fault(p, fault_type, fault_state, cycle_num)
% =========================================================================
% Injecte des défauts persistants dans les paramètres de la batterie
%
% CORRECTED:
%  - Faults are now PERSISTENT across cycles (not reset each orbit)
%  - Thermal runaway marks battery as terminal/unusable
%  - Other faults accumulate and worsen over time
%  - fault_state tracks persistence
%
% Types: 'normal', 'high_temperature', 'thermal_runaway', 'low_temperature', 'capacity_fade'
% =========================================================================

p_out = p;  % Local copy

% Initialize fault_state if not provided
if nargin < 3 || isempty(fault_state)
    fault_state.type = 'normal';
    fault_state.severity = 0.0;
    fault_state.cycle_start = 1;
    fault_state.is_terminal = false;
end

% --- Check if battery is already in terminal state (thermal runaway) ---
if fault_state.is_terminal
    % Once thermal runaway, battery is unusable for entire remaining life
    p_out.gamma = 100.0;                % Extreme aging
    p_out.alpha = 1.0e-1;               % Severe degradation rate
    return;
end

% --- Apply fault if it's the selected type OR if already in this fault state ---
if strcmp(fault_type, fault_state.type) || strcmp(fault_state.type, 'normal')
    
    % Gradually transition fault if switching types (rare; normally persistent)
    if ~strcmp(fault_type, fault_state.type) && ~strcmp(fault_state.type, 'normal')
        % Transitioning faults (could be more sophisticated)
        fault_state.severity = fault_state.severity * 0.9;  % Fade old fault
    end
    
    % Update fault state
    fault_state.type = fault_type;
    if fault_state.cycle_start == 1 && ~strcmp(fault_type, 'normal')
        fault_state.cycle_start = cycle_num;
    end
    
    % Increase severity over time once fault starts
    if cycle_num > fault_state.cycle_start
        fault_state.severity = fault_state.severity + 0.001;  % Gradual worsening
    end
    
else
    % Switch to new fault type (reset severity for new fault)
    fault_state.type = fault_type;
    fault_state.severity = 0.0;
    fault_state.cycle_start = cycle_num;
end

% --- Apply fault-specific parameter multipliers ---
switch fault_state.type
    
    case 'normal'
        % No fault: baseline aging only
        p_out.gamma = 1.0;
        fault_state.severity = 0.0;
    
    case 'high_temperature'
        % Elevated temperature operation (cooling failure, thermal imbalance)
        % Manifests as: higher resistance, faster aging
        severity_factor = 1.0 + fault_state.severity;  % Gradually worsens
        p_out.gamma = 1.5 * severity_factor;           % Faster aging
        p_out.R0 = p.R0 * (1.1 * severity_factor);     % Slightly higher resistance
        p_out.alpha = p.alpha * (1.2 * severity_factor); % Accelerated aging
    
    case 'thermal_runaway'
        % Catastrophic thermal event: battery becomes unusable
        % In real world: immediate shutdown, fire hazard, mission loss
        % In simulation: mark as terminal, extreme degradation
        p_out.gamma = 50.0;                 % Extreme aging multiplier
        p_out.alpha = 0.1;                  % Severe calendar degradation
        p_out.R0 = p.R0 * 10.0;             % Massive resistance growth
        p_out.G_structure = p.G_structure * 0.1;  % Thermal coupling degraded
        fault_state.severity = 1.0;
        fault_state.is_terminal = true;     % Mark as terminal
        
        % Optional: trigger end-of-life flag
        % In a real system, the battery would be disabled
    
    case 'low_temperature'
        % Cold operation (heater failure, insufficient heating in eclipse)
        % Manifests as: high resistance, reversible capacity loss
        severity_factor = 1.0 + fault_state.severity;
        p_out.R0 = p.R0 * (1.5 * severity_factor);  % Much higher resistance
        p_out.gamma = 0.6 * severity_factor;        % Slower aging (cold slows reactions)
        p_out.alpha = p.alpha * 0.8;                % Reduced calendar aging
    
    case 'capacity_fade'
        % Progressive capacity loss (SEI growth, active material loss)
        % Manifests as: increased aging, slightly higher resistance
        severity_factor = 1.0 + fault_state.severity;
        p_out.gamma = 2.0 * severity_factor;        % Faster aging
        p_out.alpha = p_out.alpha * (1.3 * severity_factor);  % Accelerated degradation
        p_out.R0 = p.R0 * (1.05 * severity_factor); % Slight resistance growth
    
    otherwise
        warning('Fault type unknown: %s. Using normal.', fault_type);
        p_out.gamma = 1.0;
        fault_state.type = 'normal';
end

end
