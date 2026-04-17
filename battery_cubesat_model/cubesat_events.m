function [value, isterminal, direction] = cubesat_events(t, x, p, phase)
% =========================================================================
% cubesat_events.m  —  v3  (unchanged from v2)
%
%   phase='discharge'  stop when SOC falls  to SOC_min  (direction −1)
%   phase='charge'     stop when SOC rises  to 0.99     (direction +1)
%
% Note: simulate_cycle.m defines inline event functions (evt_dch / evt_ch)
% that are the active implementations used by ode45. This file provides
% the same logic for any external caller or reference use.
% =========================================================================

SOC = x(1);

if strcmp(phase, 'discharge')
    value      = SOC - p.SOC_min;
    isterminal = 1;
    direction  = -1;   % DECREASING crossing

elseif strcmp(phase, 'charge')
    value      = SOC - 0.99;
    isterminal = 1;
    direction  = +1;   % INCREASING crossing

else
    error('cubesat_events: unknown phase "%s". Use "discharge" or "charge".', phase);
end

end
