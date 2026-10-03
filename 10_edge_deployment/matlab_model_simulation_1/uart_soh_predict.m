function soh_pred = uart_soh_predict(sp, cycle_num, window_buf)
% UART_SOH_PREDICT - send a 30x20 scaled window to the STM32 over the
% USB cable (ST-Link virtual COM port) and block until the SOH
% prediction (or a timeout) comes back.
%
% sp         : serialport object opened on the ST-Link VCP
%              (USART1 on the B-U585I-IOT02A, 921600 baud, 8N1)
% cycle_num  : uint16 cycle number, echoed back by the STM32 for a sanity check
% window_buf : 30 x 20 double matrix, ALREADY SCALED (see scale_features.m)
%
% Wire format must match App/app_uart_protocol.c on the firmware side
% exactly - see the deployment guide, Section 5.3/5.4:
%
%   Request  (this function -> STM32), 2406 bytes:
%     [0]        0xAA
%     [1]        0x55
%     [2:3]      cycle_num, uint16 LE
%     [4:2403]   window data, 600 x float32 LE, timestep-major
%     [2404:2405] CRC16/CCITT-FALSE over bytes [0:2403]
%
%   Response (STM32 -> this function), 10 bytes:
%     [0]        0xBB
%     [1]        0x66
%     [2:3]      cycle_num echo, uint16 LE
%     [4:7]      soh_pred, float32 LE
%     [8:9]      CRC16 over bytes [0:7]
%
% Returns NaN if the CRC check fails, the echoed cycle number doesn't
% match, or the read times out (matches the NaN convention your original
% cubesat_eps_main.m already uses for a failed prediction).

    REQ_HEADER  = uint8([0xAA 0x55]);
    RESP_HEADER = uint8([0xBB 0x66]);
    WINDOW_LEN  = 30;
    N_FEAT      = 20;

    %% Build request packet
    cycle_bytes = typecast(uint16(cycle_num), 'uint8');

    % Row-major = timestep-major flatten, matching the TFLite tensor's
    % natural flat layout [1,30,20]: t=0 (feat0..19), t=1 (feat0..19), ...
    flat = single(reshape(window_buf', 1, []));   % window_buf is 30x20 -> transpose then flatten row-wise
    data_bytes = typecast(flat, 'uint8');

    if numel(data_bytes) ~= WINDOW_LEN * N_FEAT * 4
        error('uart_soh_predict: window has wrong size (%d bytes, expected %d)', ...
            numel(data_bytes), WINDOW_LEN * N_FEAT * 4);
    end

    payload_for_crc = [REQ_HEADER, cycle_bytes, data_bytes];
    crc = crc16_ccitt_false(payload_for_crc);
    crc_bytes = typecast(uint16(crc), 'uint8');

    packet = [payload_for_crc, crc_bytes];

    %% Send (discard anything left over from a previous timeout first)
    flush(sp, 'input');
    write(sp, packet, 'uint8');

    %% Receive response (10 bytes, fixed size)
    RESP_LEN = 10;
    resp = read(sp, RESP_LEN, 'uint8');

    if numel(resp) ~= RESP_LEN
        warning('uart_soh_predict: cycle %d - short/timed-out response (%d bytes)', ...
            cycle_num, numel(resp));
        soh_pred = NaN;
        return;
    end

    resp = uint8(resp);

    if ~isequal(resp(1:2), RESP_HEADER)
        warning('uart_soh_predict: cycle %d - bad response header', cycle_num);
        soh_pred = NaN;
        return;
    end

    resp_cycle = typecast(resp(3:4), 'uint16');
    soh_bytes  = resp(5:8);
    resp_crc   = typecast(resp(9:10), 'uint16');

    calc_crc = crc16_ccitt_false(resp(1:8));
    if calc_crc ~= resp_crc
        warning('uart_soh_predict: cycle %d - CRC mismatch', cycle_num);
        soh_pred = NaN;
        return;
    end

    if resp_cycle ~= uint16(cycle_num)
        warning('uart_soh_predict: cycle %d - cycle echo mismatch (got %d)', ...
            cycle_num, resp_cycle);
        soh_pred = NaN;
        return;
    end

    soh_pred = double(typecast(soh_bytes, 'single'));

end


function crc = crc16_ccitt_false(bytes)
% CRC-16/CCITT-FALSE, poly 0x1021, init 0xFFFF, no reflect, no final xor.
% Must match Crc16CcittFalse() in App/app_uart_protocol.c exactly.

    crc = uint16(hex2dec('FFFF'));
    poly = uint16(hex2dec('1021'));

    for k = 1:numel(bytes)
        crc = bitxor(crc, bitshift(uint16(bytes(k)), 8));
        for b = 1:8
            if bitand(crc, uint16(hex2dec('8000'))) ~= 0
                crc = bitxor(bitshift(crc, 1), poly);
            else
                crc = bitshift(crc, 1);
            end
        end
    end

end
