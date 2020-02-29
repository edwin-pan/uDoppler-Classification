function [S,T,F] = helperDopplerSignatures(x,Tsamp)
%helperBackScatterDopplerSignature generates the micro-Doppler signatures
%from the radar return signals.
%   [SPed,SBic,Scar] = helperBackScatterDopplerSignature(xPedRec,xBicRec,xCarRec)
%   returns preprocessed micro-Doppler signatures from radar signals. Tsamp is the time interval between two waveforms.

%   Copyright 2019 The MathWorks, Inc.
    %% STFT parameters
    M = 200; % FFT window length
    beta = 6; % window parameter
    w = kaiser(M,beta); % kaiser window
    R = floor(1.7*(M-1)/(beta+1)); % ROUGH estimate
    noverlap = M-R; % overlap length
    
    [S,F,T] = stft(squeeze(sum(x,1)),1/Tsamp,'Window',w,'FFTLength',M*2,'OverlapLength',noverlap);
    S = helperPreProcess(S); % preprocessing of the spectrogram
end

function S = helperPreProcess(S)
%helperPreProcess converts each spectrogram into log-scale and normalizes each log-scale spectrogram
%to [0,1].
    
    S = 10*log10(abs(S)); % logarithmic scaling to dB
    for ii = 1:size(S,3)
        zs = S(:,:,ii);
        zs = (zs - min(zs(:)))/(max(zs(:))-min(zs(:))); % normalize amplitudes of each map to [0,1]
        S(:,:,ii) = zs;
    end
end