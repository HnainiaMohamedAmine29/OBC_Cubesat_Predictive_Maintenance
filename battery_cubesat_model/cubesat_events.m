function [value, isterminal, direction] = cubesat_events(t, x, p, phase)
% Événements pour arrêter la simulation
% phase: 'discharge' ou 'charge'

SOC = x(1);

if strcmp(phase, 'discharge')
    value = SOC - p.SOC_min;  % Arrêt quand SOC = SOC_min
elseif strcmp(phase, 'charge')
    value = SOC - 0.99;        % Arrêt quand SOC = 0.99
else
    error('Phase inconnue');
end

isterminal = 1;   % Arrêter l'intégration
direction = -1;   % Pour décharge: SOC décroît; pour charge: SOC croît est OK aussi
end

% Fonctions wrapper pour les événements séparés
function [val, ist, dir] = evt_dch_wrapper(t, x, p)
    [val, ist, dir] = cubesat_events(t, x, p, 'discharge');
end

function [val, ist, dir] = evt_ch_wrapper(t, x, p)
    [val, ist, dir] = cubesat_events(t, x, p, 'charge');
end