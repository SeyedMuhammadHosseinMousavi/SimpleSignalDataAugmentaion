function AugmentedSample = ApplyDataAug(originalSample,TimeShiftRange,ScaleRange,NoiseMagnitude)

% Time shifting
timeShift = (2*rand - 1) * TimeShiftRange;
Shifted = circshift(originalSample, round(timeShift));
AugmentedSample=Shifted;

% Amplitude scaling
AmplitudeScale = 0.9 + (2*rand - 1) * ScaleRange;
Scaled = Shifted * AmplitudeScale;
AugmentedSample=Scaled;

% Random Gaussian noise
Noise = NoiseMagnitude * randn(size(originalSample));
AugmentedSample = AugmentedSample + Noise;

end

