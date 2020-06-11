function [impulse_train, f0maps] = generate_impulse_train(f0, fs, hop_size)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This function generates an impulse train as per the pitch contour f0. 
%
% INPUTS: f0: pitch contour
%         fs: sampling frequency
%         hop_size: size of each pitch window in seconds
% 
% OUTPUTS: impulse_train: The output impulse train where the frequency of
%          1s occuring in the vector is the instantaneous frequency in f0.
%
% Written by: Devansh Zurale
% Last date modified: 06/02/2020
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

f0Len = length(f0);
hop_len = hop_size*fs;
it_len = ceil(f0Len*hop_size*fs);

count = 1;
impulse_train = zeros(it_len, 1);
f0maps = zeros(it_len, 1);
eveoddflag = true;
for i = 1:f0Len
    thisf0 = f0(i);
    if thisf0
        thisT0 = 1/thisf0;
        thisN0 = round(fs*thisT0);
        thisFrameStart = hop_len*(i-1)+1;
        while count < i*hop_len
            impulse_train(count) = 1;
            f0maps(count) = thisf0;
            count = count + thisN0;
        end
    else
        if eveoddflag
            count = count + floor(hop_len);
        else 
            count = count + ceil(hop_len);
        end
        eveoddflag = ~eveoddflag;
    end
end