function scaled_row = scale_features(feat_struct, ordered_features, mean_vec, scale_vec)
% SCALE_FEATURES - build a 1x20 raw feature row from feat_struct in the
% correct model order, then apply the StandardScaler transform.
%
% feat_struct       : struct from compute_features(), with SOH_lag_1 patched
%                      in by cubesat_eps_main_stm32.m (same as your original loop)
% ordered_features  : cell array of 20 names, the model's required order
% mean_vec/scale_vec: from scaler_constants()

n = numel(ordered_features);
raw_row = zeros(1, n);

for i = 1:n
    key = ordered_features{i};
    if ~isfield(feat_struct, key)
        error("scale_features: missing feature '%s'", key);
    end
    val = feat_struct.(key);
    if isnan(val) || isinf(val)
        val = 0.0;
    end
    raw_row(i) = val;
end

scaled_row = (raw_row - mean_vec) ./ scale_vec;

end
