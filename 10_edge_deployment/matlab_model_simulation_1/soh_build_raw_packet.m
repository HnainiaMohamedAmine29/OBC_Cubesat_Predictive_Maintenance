function packet = soh_build_raw_packet(cycle_num, row, soh_lag_ext, use_ext_lag)
% Build the 104-byte v2 request: ONLY the raw simulation row, no features, no scaling.
%
%   [0]=0xAA [1]=0x56 [2:3]=cycle u16 [4]=flags u8 [5]=0
%   [6:93]  11 x float64 LE: V_mean_V V_min_V V_max_V Tavg_C Tmin_C Tmax_C QD_Ah QC_Ah DoD discharge_time_min T_amb_K
%   [94:101] soh_lag_ext float64 (only used by the board when flags bit0 = 1)
%   [102:103] CRC16/CCITT-FALSE over bytes [0:101]
%
% row : the struct returned by simulate_cycle()
    if nargin < 3 || isempty(soh_lag_ext), soh_lag_ext = 0; end
    if nargin < 4 || isempty(use_ext_lag), use_ext_lag = false; end

    vals = double([row.V_mean_V, row.V_min_V, row.V_max_V, ...
                   row.Tavg_C,   row.Tmin_C,  row.Tmax_C, ...
                   row.QD_Ah,    row.QC_Ah,   row.DoD, ...
                   row.discharge_time_min, row.T_amb_K]);

    flags = uint8(0);
    if use_ext_lag, flags = uint8(1); end

    body = [uint8([170 86]), ...                         % 0xAA 0x56
            typecast(uint16(cycle_num), 'uint8'), ...
            flags, uint8(0), ...
            typecast(vals, 'uint8'), ...
            typecast(double(soh_lag_ext), 'uint8')];
    packet = [body, typecast(soh_crc16(body), 'uint8')];

    if numel(packet) ~= 104
        error('soh_build_raw_packet: packet is %d bytes, expected 104', numel(packet));
    end
end
