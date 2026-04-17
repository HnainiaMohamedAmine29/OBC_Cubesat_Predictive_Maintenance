function [p_out, fault_state] = inject_fault(p, fault_type, fault_state, cycle_num)
% =========================================================================


p_out     = p;
p_out.OCV = p.OCV;   % preserve function handle

% --- Initialise ---
if nargin < 3 || isempty(fault_state)
    fault_state.type        = 'normal';
    fault_state.severity    = 0.0;
    fault_state.cycle_start = 1;
    fault_state.is_terminal = false;
end

% --- Terminal battery ---
if fault_state.is_terminal
    p_out.gamma = 100.0;
    p_out.alpha = 1.0e-1;
    return;
end

% --- Severity update ---
if strcmp(fault_type, fault_state.type)
    fault_state.severity = min(fault_state.severity + 0.001, 3.0);
else
    fault_state.type        = fault_type;
    fault_state.severity    = 0.0;
    fault_state.cycle_start = cycle_num;
end

sf = 1.0 + fault_state.severity;

% --- Parameter modifications by fault type ---
switch fault_state.type

    case 'normal'
        p_out.gamma      = 1.0;
        fault_state.severity = 0.0;

    case 'high_temperature'
        p_out.gamma       = 1.5 * sf;
        p_out.R0          = p.R0 * (1.10 * sf);
        p_out.alpha       = p.alpha * (1.20 * sf);
        p_out.T_amb_sun   = min(p.T_operating_max, ...
                                p.T_amb_sun + 10 * fault_state.severity);

    case 'low_temperature'
        p_out.R0          = p.R0 * (1.5 * sf);
        p_out.gamma       = 0.6 * sf;
        p_out.alpha       = p.alpha * 0.8;
        p_out.P_heater    = 0.0;               % heater failed

    case 'capacity_fade'
        p_out.gamma = 2.0 * sf;
        p_out.alpha = p.alpha * (1.3 * sf);
        p_out.R0    = p.R0 * (1.05 * sf);

    case 'thermal_runaway'
        p_out.gamma               = 50.0;
        p_out.alpha               = 0.10;
        p_out.R0                  = p.R0 * 10.0;
        p_out.G_structure         = p.G_structure * 0.1;
        fault_state.severity      = 1.0;
        fault_state.is_terminal   = true;

    otherwise
        warning('inject_fault: unknown fault "%s". Using normal.', fault_type);
        p_out.gamma      = 1.0;
        fault_state.type = 'normal';
end

end
