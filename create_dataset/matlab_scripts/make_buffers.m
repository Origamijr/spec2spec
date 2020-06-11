function buffer = make_buffers(in, win_len, hop_len)

in_len = length(in);

eveoddflag = true;
start_pos = 1;

buffer = [];
while start_pos <= in_len
    end_pos = start_pos+win_len-1;
    if end_pos <= in_len
        thisbuf = in(start_pos:end_pos);
    else
        diff = end_pos - in_len;
        thisbuf = [in(start_pos:end); zeros(diff, 1)];
    end
    
    if eveoddflag
        start_pos = start_pos+floor(hop_len);
    else
        start_pos = start_pos+ceil(hop_len);
    end
    eveoddflag = ~eveoddflag;
    buffer = [buffer, thisbuf];
end

end
        
