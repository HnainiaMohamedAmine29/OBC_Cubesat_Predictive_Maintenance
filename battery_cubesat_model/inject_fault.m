function p_out = inject_fault(p, fault_type)
% Injecte des défauts dans les paramètres de la batterie
% NOTE: Ne modifie PAS T_amb (température venant du cycle LEO)
% Types: 'normal', 'high_temperature', 'thermal_runaway', 'low_temperature', 'capacity_fade'

p_out = p;  % Copie locale

switch fault_type
    case 'normal'
        p_out.gamma = 1.0;

    case 'high_temperature'
        p_out.gamma = 2.0;           % Vieillissement accéléré
        p_out.R0 = p_out.R0 * 1.2;   % Résistance interne un peu plus haute

    case 'thermal_runaway'
        p_out.h_conv = 0.8;       % Très faible refroidissement (conduction)
        p_out.R0 = p_out.R0 * 4.0; % Résistance interne x4
        p_out.gamma = 6.0;        % Vieillissement extrême

    case 'low_temperature'
        p_out.R0 = p_out.R0 * 1.8;  % Résistance interne élevée (effet batterie froide)
        p_out.gamma = 0.7;          % Vieillissement légèrement réduit (moins réactions)

    case 'capacity_fade'
        p_out.gamma = 2.5;
        p_out.alpha = p_out.alpha * 1.4;  % Accélération de dégradation

    otherwise
        warning('Fault type unknown: %s. Using normal.', fault_type);
end
end