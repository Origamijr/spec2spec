function glottal_flow = generate_glottal_flow(impulse_train, fs, f0maps)

it_len = length(impulse_train);
glottal_flow = zeros(it_len, 1);
for i = 1:it_len
    if impulse_train(i)
        thisf0 = f0maps(i);
        thisT0 = 1/thisf0;
        thisN0 = thisT0*fs;
        thisgp_len = floor(0.8*thisN0);
        glottal_pulse = hann(thisgp_len);
        glottal_flow(i:i+thisgp_len-1) = glottal_pulse;
    end
end

end