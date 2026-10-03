function r = soh_parse_raw_response(resp, cycle_num)
% Parse the 24-byte v2 response.
%   [0]=0xBB [1]=0x67 [2:3]=cycle u16 [4]=status u8 [5]=0 [6:9]=soh f32
%   status: 0 OK, 1 DUPLICATE, 2 GAP, 3 INFER_ERR, 4 WARMUP (nominal 1.0, model not run)
%   [10:13]=t_preproc_us u32 [14:17]=t_infer_us u32 [18:21]=arena_used u32 [22:23]=CRC16
% r.ok is false (and r.soh = NaN) on any framing / CRC / echo error.
    r = struct('ok', false, 'soh', NaN, 'status', uint8(255), 't_pre_us', NaN, 't_inf_us', NaN, 'arena', NaN, 'why', '');
    resp = uint8(resp(:)');
    if numel(resp) ~= 24,                              r.why = 'short/timed-out response'; return; end
    if ~isequal(resp(1:2), uint8([187 103])),          r.why = 'bad header';               return; end
    if soh_crc16(resp(1:22)) ~= typecast(resp(23:24), 'uint16'), r.why = 'CRC mismatch';  return; end
    if typecast(resp(3:4), 'uint16') ~= uint16(cycle_num),       r.why = 'cycle echo mismatch'; return; end
    r.status   = resp(5);
    r.soh      = double(typecast(resp(7:10),  'single'));
    r.t_pre_us = double(typecast(resp(11:14), 'uint32'));
    r.t_inf_us = double(typecast(resp(15:18), 'uint32'));
    r.arena    = double(typecast(resp(19:22), 'uint32'));
    r.ok       = (r.status ~= 3);          % 3 = inference error on the board
    if ~r.ok, r.soh = NaN; r.why = 'inference error on board'; end
end
