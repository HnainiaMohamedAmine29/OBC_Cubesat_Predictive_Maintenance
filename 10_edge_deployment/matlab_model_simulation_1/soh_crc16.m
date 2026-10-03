function crc = soh_crc16(bytes)
% CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection, no final xor.
% Must match Crc16CcittFalse() in App/app_uart_protocol.c.
    crc  = uint16(65535);          % 0xFFFF
    poly = uint16(4129);           % 0x1021
    top  = uint16(32768);          % 0x8000
    for k = 1:numel(bytes)
        crc = bitxor(crc, bitshift(uint16(bytes(k)), 8));
        for b = 1:8
            if bitand(crc, top) ~= 0
                crc = bitxor(bitshift(crc, 1), poly);
            else
                crc = bitshift(crc, 1);
            end
        end
    end
end
