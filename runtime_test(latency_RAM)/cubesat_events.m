function [value, isterminal, direction] = cubesat_events(t, x, p, phase)
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
