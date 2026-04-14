function [value, isterminal, direction] = cubesat_events(t, x, p, phase)
% =========================================================================
% Event detection for battery cycle termination
% CORRECTED: Unified implementation, consistent direction values
%
% phase: 'discharge' or 'charge'
% =========================================================================

SOC = x(1);

if strcmp(phase, 'discharge')
    % Stop discharge when SOC falls to minimum
    value = SOC - p.SOC_min;      % Zero crossing when SOC = SOC_min
    isterminal = 1;               % Stop integration
    direction = -1;               % Detect decreasing (SOC declining during discharge)
    
elseif strcmp(phase, 'charge')
    % Stop charge when SOC reaches 99%
    value = SOC - 0.99;           % Zero crossing when SOC = 0.99
    isterminal = 1;               % Stop integration
    direction = 1;                % Detect increasing (SOC rising during charge)
    
else
    error('Unknown phase: %s (use "discharge" or "charge")', phase);
end

end
