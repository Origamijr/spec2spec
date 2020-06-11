function psd = getpsd(buffer_in, window, nfft)

window_buffer = buffer_in.*window;
f = abs(fft(window_buffer, nfft));
f = f(1:nfft/2+1, :);

psd = f.^2;
psd(2:end-1, :) = psd(2:end-1, :)*2;
psd = psd/nfft;



end