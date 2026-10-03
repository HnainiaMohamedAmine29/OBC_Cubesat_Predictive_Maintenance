function window_buf = build_window(scaled_row, window_buf, window_len)
% BUILD_WINDOW - maintain a persistent [window_len x n_features] buffer.
%
% Warm-up rule: on the very first call (empty window_buf), the whole
% window is seeded with window_len copies of the first scaled row, so
% the model gets a full 30-timestep window - and therefore a real
% prediction - starting at cycle 1, instead of returning nothing for the
% first 29 cycles.
%
% Every subsequent call slides the window: drop the oldest row, append
% the new one.
%
% Usage (call once per cycle, in order):
%   window_buf = build_window(scaled_row, window_buf, 30);
%
% Pass window_buf = [] on the first call of a fresh run.

    if isempty(window_buf)
        % Cycle 1: seed the entire window with the first sample.
        window_buf = repmat(scaled_row, window_len, 1);
    else
        % Slide: drop oldest row, append new row.
        window_buf = [window_buf(2:end, :); scaled_row];
    end

end
