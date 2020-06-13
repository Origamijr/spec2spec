function mfsc = spec2melspec(spectrogram, fs, fRange, noMelChannels)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the mel spectrogram given the input of a spectrogram. Whether the
% function returns a power mel spectrogram, or magnitude spectrogram
% depends on the spectrogram input itself.
%
% INPUTS: spectrogram: A onesided spectrogram
%         fs: sampling frequency
%         fRange: A 2 element vector [fLow, fHigh] where fLow is the lowest
%                 frequency in the mel scale and fHigh is the highest 
%                 frequency in the mel scale
%         noMelChannels: No of mel channels
%
% OUTPUTs: mfsc: The mel frequency spectral coefficients
%
% Written by: Devansh Zurale
% Last updated: June 6th 2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Making the Mel and Frequency scales
fLow = fRange(1);
fHigh = fRange(2);
nfft = (size(spectrogram, 1)-1)*2;
noWindows = size(spectrogram, 2);
mLow = (2595*log10(1+fLow/700))';
mHigh = (2595*log10(1+fHigh/700))';
mDiff = (mHigh - mLow)/(noMelChannels+1);

for i = 1:noMelChannels+2
    mScale(i,1) = mLow + (i-1)*mDiff;
end

fScale = 700*(10.^(mScale/2595) - 1);
wScale = fScale/fs*2;
fftScale = floor(fScale*nfft/fs) + 1;

% Creating the filter bank

filterbank = [];
for i = 1:noMelChannels
    start = zeros(1, fftScale(i) - 1);
    rise = linspace(0, 1, fftScale(i+1) - fftScale(i) + 1);
    fall = linspace(1, 0, fftScale(i+2)-fftScale(i+1) + 1);
    finish = zeros(1, nfft/2-fftScale(i+2)+1);
    thisFilter = [start, rise, fall(2:end), finish];
    %thisfFilter = [thisfFilter';flipud((thisfFilter(2:end-1))')];
    filterbank = [filterbank, thisFilter'];
end

% Computing mfsc
mfsc = [];
for i = 1:noWindows
    thisSpectrum = spectrogram(:, i);
    melEnergy = thisSpectrum.*filterbank;
    thismfsc = sum(melEnergy, 1);
    mfsc = [mfsc, thismfsc'];
end

end