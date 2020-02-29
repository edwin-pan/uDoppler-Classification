function [xPedRecF,xBicRecF,xCarRecF,Tsamp] = helperBackScatterSignals(numPed,numBic,numCar)
%helperBackScatterSignals generates the radar return signals from objects.
%   [xPedRec,xBicRec,xCarRec] = helperBackScatterSignals(numPed,numBic,numCar)
%   generates radar return signals according to specific object numbers. 
%   numPed, numBic, numCar are the numbers of pedestrians, bicyclists, and cars,
%   respectively. Tsamp is the time interval between two waveforms.

%   Copyright 2019 The MathWorks, Inc.
%%
    rng(1) % initialize the random generator to ensure results are reproducible.
    
    % radar operating related parameters
    bw = 250e6; % waveform bandwidth
    fs = bw*2; % waveform sampling frequency
    c = 3e8;                
    fc = 24e9; % waveform carrier frequency
    tm = 1e-6; % waveform repetition time                
    
    % radar plat parameters
    radar_pos = [0;0;0]; % radar position
    radar_vel = [0;0;0]; % radar velocity
    radarplat = phased.Platform('InitialPosition',radar_pos,'Velocity',radar_vel,...
        'OrientationAxesOutputPort',true);
    
    % radar waveform
    wav = phased.FMCWWaveform('SampleRate',fs,'SweepTime',tm,'SweepBandwidth',bw);
    tx = phased.Transmitter('PeakPower',1,'Gain',25);
    txWave = tx(wav());
    
    % Simulation setup
    maxbicSpeed = 10; % maximum speed of the Bicyclist
    lambda = c/fc;
    oversamplingFactor = 1.11; % oversampling factor in frequency domain
    fmax = (2*maxbicSpeed/lambda)*oversamplingFactor; % calculate the sampling frequency based on the maximum speed of a bicyclist
    Tsamp = 1/(2*fmax); % slow time sampling interval
    timeDuration = 2; % simulation time duration in seconds
    npulse = floor(timeDuration/Tsamp); % number of pulses
    [posr,velr,~] = radarplat(Tsamp); % radar position and velocity
        
    % signal initialization
    xPedRec = complex(zeros(round(fs*tm),npulse));
    xBicRec = complex(zeros(round(fs*tm),npulse));
    xCarRec = complex(zeros(round(fs*tm),npulse));
    
    xPedRecF = complex(zeros(size(xPedRec,1),size(xPedRec,2),numPed));
    xBicRecF = complex(zeros(size(xBicRec,1),size(xBicRec,2),numBic));
    xCarRecF = complex(zeros(size(xCarRec,1),size(xCarRec,2),numCar));
    
    % area of interest 
    yLocLimit = [-10,10];
    xLocLimit = [5,45];
    
    fprintf("here pedestrian\n")
    % Generation of pedestrian signals    
    for ii = 1:numPed
        % Pedestrian parameters
        ped_pos = [xLocLimit(1) + (xLocLimit(2)-xLocLimit(1))*rand;
            yLocLimit(1) + (yLocLimit(2)-yLocLimit(1))*rand;
            0]; % initial location
        ped_height = 1.5 + (2-1.5)*rand; % height in U[1.5,2] meters
        ped_speed = rand*ped_height*1.4; % speed in U[0,1.4*height] m/s
        ped_heading = -180 + 360*rand; % heading in U[-180,180] degrees
        
        pedestrian = backscatterPedestrian('InitialPosition',ped_pos,...
            'InitialHeading',ped_heading,'PropagationSpeed',c,...
            'OperatingFrequency',fc,'Height',ped_height,...
            'WalkingSpeed',ped_speed); % pedestrian object
        channel_ped = phased.FreeSpace('PropagationSpeed',c,'OperatingFrequency',fc,...
            'TwoWayPropagation',true,'SampleRate',fs); % channel
        
        for m = 1:npulse
            [posPed,velPed,axPed] = move(pedestrian,Tsamp,ped_heading); % pedestrian moves
            [~,angrPed] = rangeangle(posr,posPed,axPed); % propagation path direction
            xPedCh = channel_ped(repmat(txWave,1,size(posPed,2)),posr,posPed,velr,velPed); % simulate channel
            xPed = reflect(pedestrian,xPedCh,angrPed); % signal reflection
            xPedRec(:,m) = xPed; % received m-th pulse            
        end
        
        xPedRecF(:,:,ii) = conj(dechirp(xPedRec,txWave)); % convert received signals to baseband
        
    end
    
    fprintf("here bicyclist\n")
    % Generation of bicyclist signals
    for ii = 1:numBic  
        % Bicyclist parameters
        bic_pos = [xLocLimit(1) + (xLocLimit(2)-xLocLimit(1))*rand;
            yLocLimit(1) + (yLocLimit(2)-yLocLimit(1))*rand;
            0]; % initial location
        bicyclistSpeed = 1 + (10-1)*rand; % Speed in U[1,10] meters
        bic_heading = -180 + 360*rand; % heading in U[-180,180] degrees
        GearTransmissionRatio = 0.5 + (6-0.5)*rand; % in U[0.5,6]
        NumWheelSpokes = 36; % number of spokes
        Coast = rand<0.5; % 50% chance to be pedaling or coasting
        bicyclist = backscatterBicyclist('InitialPosition',bic_pos,'InitialHeading',bic_heading,...
            'Speed',bicyclistSpeed,'PropagationSpeed',c,'OperatingFrequency',fc,...
            'GearTransmissionRatio',GearTransmissionRatio,'NumWheelSpokes',NumWheelSpokes,...
            'Coast',Coast); % bicyclist object
        channel_bic = phased.FreeSpace('PropagationSpeed',c,'OperatingFrequency',fc,...
            'TwoWayPropagation',true,'SampleRate',fs);
        
        for m = 1:npulse
            [posBic,velBic,axBic] = move(bicyclist,Tsamp,bic_heading); % pedestrian moves
            [~,angrBic] = rangeangle(posr,posBic,axBic); % propagation path direction
            xBicCh = channel_bic(repmat(txWave,1,size(posBic,2)),posr,posBic,velr,velBic); % simulate channel
            xBic = reflect(bicyclist,xBicCh,angrBic); % signal reflection
            xBicRec(:,m) = xBic; % received m-th pulse
        end
      
        xBicRecF(:,:,ii) = conj(dechirp(xBicRec,txWave)); % convert received signals to baseband
    end
    
    fprintf("here car\n")
    maxCarSpeed = 10;
    % Generation of car signals
    for ii = 1:numCar
        % Car parameters
        car_pos = [xLocLimit(1) + (xLocLimit(2)-xLocLimit(1))*rand;
            yLocLimit(1) + (yLocLimit(2)-yLocLimit(1))*rand;
            0]; % initial location
        car_vel = [-maxCarSpeed+(maxCarSpeed+maxCarSpeed)*rand;
            -maxCarSpeed+(maxCarSpeed+maxCarSpeed)*rand;
            0]; % car velocity
        car = phased.Platform('InitialPosition',car_pos,'Velocity',car_vel,...
            'OrientationAxesOutputPort',true); % car object
        carTgt = phased.RadarTarget('PropagationSpeed',c,'OperatingFrequency',fc,'MeanRCS',10);
        chan_car = phased.FreeSpace('PropagationSpeed',c,'OperatingFrequency',fc,...
            'TwoWayPropagation',true,'SampleRate',fs);
        
        for m = 1:npulse
            [posCar,velCar,~] = car(Tsamp); % pedestrian moves
            xCarCh = chan_car(repmat(txWave,1,size(posCar,2)),posr,posCar,velr,velCar); % simulate channel
            xCar = carTgt(xCarCh); % detection
            xCarRec(:,m) = xCar; % received m-th pulse
        end
        
        xCarRecF(:,:,ii) = conj(dechirp(xCarRec,txWave)); % convert received signals to baseband
    end
end