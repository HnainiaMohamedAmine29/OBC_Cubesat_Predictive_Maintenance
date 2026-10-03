function r = uart_soh_predict_raw(sp, cycle_num, row, soh_lag_ext, use_ext_lag, n_retry)
% Send ONE raw simulation row to the STM32 (v2 protocol) and wait for the SOH.
% The board computes the features, scales, builds the 30-cycle window and runs the model.
%
% sp          : serialport on the ST-Link VCP (921600 baud, 8N1)
% row         : struct from simulate_cycle()  (raw data only)
% soh_lag_ext : only used if use_ext_lag = true (parity test against the old pipeline)
% n_retry     : resend the same cycle if the response is lost (board answers DUPLICATE without advancing state)
    if nargin < 4, soh_lag_ext = 0; end
    if nargin < 5, use_ext_lag = false; end
    if nargin < 6, n_retry = 2; end

    packet = soh_build_raw_packet(cycle_num, row, soh_lag_ext, use_ext_lag);
    for attempt = 0:n_retry
        flush(sp, 'input');
        write(sp, packet, 'uint8');
        resp = read(sp, 24, 'uint8');
        r = soh_parse_raw_response(resp, cycle_num);
        if r.ok, return; end
        warning('uart_soh_predict_raw: cycle %d attempt %d: %s', cycle_num, attempt + 1, r.why);
    end
end
