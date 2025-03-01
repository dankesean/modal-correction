%% Modal error analysis code - RC for compensation
% Author - Sean McGowan, University of Adelaide
% Date - 6/5/24

clear all
close all
clc

%% Load data and split into training and testing

load('gulfstream_data.mat')

Tspan_train = 1:(9*366+27*365);
Tspan_test = 1:(9*366+28*365); 

SST_sat = SST_prediction_sat(:,:,Tspan_train);
SST_model = SST_prediction_model(:,:,Tspan_train);

%% Plot satellite and model data

Tlim_full = [floor(min(SST_sat,[],'all')) ceil(max(SST_sat,[],'all'))];

figure
subplot(1,2,1)
h1 = imagesc(lat_sat(indexX(1):indexX(2)),lon_sat(indexY(1):indexY(2)),zeros(50,50));
xlim(lat_sat([indexX(1) indexX(2)-1]))
ylim(lon_sat([indexY(1) indexY(2)-1]))
set(gca,'Xdir','reverse','Ydir','reverse')
view(270,270)
colormap turbo
clim(Tlim_full)
title('Satellite data')

subplot(1,2,2)
h2 = imagesc(lat_sat(indexX(1):indexX(2)),lon_sat(indexY(1):indexY(2)),zeros(50,50));
xlim(lat_sat([indexX(1) indexX(2)-1]))
ylim(lon_sat([indexY(1) indexY(2)-1]))
set(gca,'Xdir','reverse','Ydir','reverse')
view(270,270)
colormap turbo
clim(Tlim_full)
title('Model data')
    
for i=length(Tspan_train)-365:length(Tspan_train)
    set(h1,'cdata',SST_sat(:,:,i)) 
    set(h2,'cdata',SST_model(:,:,i)) 
    
    sgtitle(datestr(datetime(1985,1,1)+i-1))
    
    drawnow;
    pause(0.01);
end

%% Plot discrepancy

SST_disc = SST_sat-SST_model;

Tlim_disc = [floor(min(SST_disc,[],'all')) ceil(max(SST_disc,[],'all'))];

figure
h1 = imagesc(lat_sat(indexX(1):indexX(2)),lon_sat(indexY(1):indexY(2)),zeros(50,50));
xlim(lat_sat([indexX(1) indexX(2)-1]))
ylim(lon_sat([indexY(1) indexY(2)-1]))
set(gca,'Xdir','reverse','Ydir','reverse')
view(270,270)
colorbar
colormap turbo
clim(Tlim_disc)
    
for i=length(Tspan_train)-365:length(Tspan_train)
    set(h1,'cdata',SST_disc(:,:,i)) 
    
    sgtitle(datestr(datetime(1985,1,1)+i-1))
    
    drawnow;
    pause(0.01);
end

%% Calculate EOFs and corresponding principal component time series

Snapshots = permute(SST_disc,[3 1 2]);
Snapshots(isnan(Snapshots))=0;
dims = size(Snapshots);

Snapshots = (reshape(Snapshots,dims(1),prod(dims(2:end)))).';

Neof = 40;

[EOF,lambda] = eig(Snapshots*Snapshots'); % eigenvectors of covariance matrix are EOFs
lambda = diag(lambda);

[~,index] = sort(abs(lambda),'descend'); % order by eigenvalue modulus
lambda = diag(lambda(index));
EOF = EOF(:,index);

% project data onto EOFs to get time series
PCTS = Snapshots'*EOF(:,1:Neof); % principal component time series

covExp = diag(lambda)/trace(lambda); % covariance explained by each EOF
sum(covExp(1:Neof)) % total covariance explained by chosen EOFs


%% Plot EOFs

figure
figuresize(25,12)
for f = 1:8
subplot(2,4,f)
imagesc(lat_sat(indexX(1):indexX(2)),lon_sat(indexY(1):indexY(2)),reshape(EOF(:,f),dims(2:3)))
xlim(lat_sat([indexX(1) indexX(2)-1]))
ylim(lon_sat([indexY(1) indexY(2)-1]))
set(gca,'Xdir','reverse','Ydir','reverse')
view(270,270)
grid off
colormap turbo
figtitle = ['$R^2 = ', num2str(covExp(f)*100), ' \%$'];
title(figtitle,'Interpreter','latex' );
if f<5
    yticklabels('')
else
    degreetick y
end
if f==1 || f==5
    degreetick x
else
    xticklabels('')
end
end
s = sky(1000);
colormap(gcf,s(:,[3 1 1]));

%% EOF reconstruction 

figure
subplot(1,2,1);
h1 = imagesc(lat_sat(indexX(1):indexX(2)),lon_sat(indexY(1):indexY(2)),zeros(50,50));
xlim(lat_sat([indexX(1) indexX(2)-1]))
ylim(lon_sat([indexY(1) indexY(2)-1]))
set(gca,'Xdir','reverse','Ydir','reverse')
view(270,270)
title('True')
colorbar
colormap turbo
clim(Tlim_disc)

subplot(1,2,2)
h2 = imagesc(lat_sat(indexX(1):indexX(2)),lon_sat(indexY(1):indexY(2)),zeros(50,50));
xlim(lat_sat([indexX(1) indexX(2)-1]))
ylim(lon_sat([indexY(1) indexY(2)-1]))
set(gca,'Xdir','reverse','Ydir','reverse')
view(270,270)
title('RC reconstruction')
colorbar
colormap turbo
clim(Tlim_disc)

for i = length(Tspan_train)-365:length(Tspan_train)
    % Hankel DMD reconstruction
    SST_recon_RC = PCTS(i,:)*EOF(:,1:Neof)';
    
    % reshape back to original dimensions
    SST_recon_RC = reshape(SST_recon_RC,dims(2:3));

    set(h2,'cdata',SST_recon_RC)
    set(h1,'cdata',SST_disc(:,:,Tspan_train(i)))
    
    drawnow;
    pause(0.01);
end


%% Echo state network
% adapted from https://mantas.info/code/simple_esn/ by Mantas Lukosevicius
% number of realisations
Nreal = 20;

% training and testing data lengths
Ttrain = length(Tspan_train)-1;
Ttest = length(Tspan_test)-length(Tspan_train);
% reservoir initialisation length
Tinit = 1000; 

% reservoir parameters
Ninput = Neof;
Noutput = Neof;
Nres = 5000;
alpha = 1; % leaking rate

Ypredict = zeros(Noutput, Ttest, Nreal);

for nn = 1:Nreal
% input weights
Win = (rand(Nres,Ninput)-0.5) .* (4*10^-1);

% sparse reservoir weights
W = sprand(Nres,Nres,3/Nres); 

% normalizing and setting spectral radius
rhoW = abs(eigs(W,1,'LM'));
W = W .* (0.1/rhoW);

% reservoir states
R = zeros(Nres,Ttrain-Tinit-1);
% output states
Y = PCTS(Tinit+2:Ttrain,:)';

% run the reservoir with the data
r = zeros(Nres,1);
for t = 1:Ttrain-1
	u = PCTS(t,:)';
	r = (1-alpha)*r+alpha*tanh(Win*u+W*r);
    r(1:2:end,:) = r(1:2:end,:).^2; % nonlinear transformation
	if t > Tinit
		R(:,t-Tinit) = r;
	end
end

% train the output by ridge regression
reg = 10^-4;  % regularization coefficient
Wout = ((R*R' + reg*eye(Nres)) \ (R*Y'))'; 

% run the trained ESN in a generative mode. no need to initialize here, 
% because x is initialized with training data and we continue from there.
u = PCTS(Ttrain,:)';
for t = 1:Ttest 
	r = (1-alpha)*r+alpha*tanh(Win*u+W*r);
    r(1:2:end,:) = r(1:2:end,:).^2; % nonlinear transformation
	y = Wout*r;
	Ypredict(:,t,nn) = y;
	u = y;
end
end

Ypredictmean = squeeze(nanmean(Ypredict,3));

figure
plot(1:4*365+length(Ttest),[PCTS(end-3*365:end,:)' Ypredictmean])
xline(3*365+length(Ttest))

%% Plot results

figure
subplot(1,4,1);
h1 = imagesc(lat_sat(indexX(1):indexX(2)),lon_sat(indexY(1):indexY(2)),zeros(50,50));
xlim(lat_sat([indexX(1) indexX(2)-1]))
ylim(lon_sat([indexY(1) indexY(2)-1]))
set(gca,'Xdir','reverse','Ydir','reverse')
view(270,270)
grid off
title('True')
colormap turbo
clim(Tlim_full)

subplot(1,4,2);
h2 = imagesc(lat_sat(indexX(1):indexX(2)),lon_sat(indexY(1):indexY(2)),zeros(50,50));
xlim(lat_sat([indexX(1) indexX(2)-1]))
ylim(lon_sat([indexY(1) indexY(2)-1]))
set(gca,'Xdir','reverse','Ydir','reverse')
view(270,270)
grid off
title('Model')
colormap turbo
clim(Tlim_full)

subplot(1,4,3)
h3 = imagesc(lat_sat(indexX(1):indexX(2)),lon_sat(indexY(1):indexY(2)),zeros(50,50));
xlim(lat_sat([indexX(1) indexX(2)-1]))
ylim(lon_sat([indexY(1) indexY(2)-1]))
set(gca,'Xdir','reverse','Ydir','reverse')
view(270,270)
grid off
title('RC compensated model')
colormap turbo
clim(Tlim_full)

h = colorbar;
set(h,'position',[0.8,0.1,0.02,0.75])
h.Label.String = 'T (K)';
h.Label.FontSize = 12;

predict_length = 90;

E_model = [];
E_RC = [];
E_spatial_model = zeros(50,50,predict_length+1);
E_spatial_H = zeros(50,50,predict_length+1);
% prediction for half a year
for i = length(Tspan_train):length(Tspan_train)+predict_length
    % RC reconstruction
    SST_recon_RC = Ypredictmean(:,i-length(Tspan_train)+1)'*EOF(:,1:Neof)';
    
    % reshape back to original dimensions
    SST_recon_RC = reshape(SST_recon_RC,dims(2:3))+SST_prediction_model(:,:,i);
    
    set(h3,'cdata',SST_recon_RC)
    set(h2,'cdata',SST_prediction_model(:,:,i))
    set(h1,'cdata',SST_prediction_sat(:,:,i))
    
    E_model = [E_model (1/prod(dims(2:end))*sum((SST_prediction_model(:,:,i)-SST_prediction_sat(:,:,i)).^2,'all'))^0.5];
    E_RC = [E_RC (1/prod(dims(2:end))*sum((SST_recon_RC-SST_prediction_sat(:,:,i)).^2,'all'))^0.5];
    
    E_spatial_model(:,:,i-length(Tspan_train)+1) = (SST_prediction_model(:,:,i)-SST_prediction_sat(:,:,i)).^2;
    E_spatial_H(:,:,i-length(Tspan_train)+1) = (SST_recon_RC-SST_prediction_sat(:,:,i)).^2;
    
    drawnow;
    pause(0.05);
end

figure
plot(E_model,'color',[0.1855 0.4989 0.2322])
hold on
plot(E_RC,'r')
ylabel('Spatially averaged error','interpreter','latex')
xlabel('Time (days)','interpreter','latex')
xlim([0 90])

RMSE_model = sqrt((1./(1:length(E_model))).*cumsum(E_model.^2));
RMSE_RC = sqrt((1./(1:length(E_RC))).*cumsum(E_RC.^2));

figure
plot(RMSE_model,'color',[0.1855 0.4989 0.2322])
hold on
plot(RMSE_RC,'r')
ylabel('RMSE(t)','interpreter','latex')
xlabel('Time (days)','interpreter','latex')
xlim([0 90])

E_spatial_model_avg = sqrt(1/(predict_length+1)*sum(E_spatial_model,3));
E_spatial_H_avg = sqrt(1/(predict_length+1)*sum(E_spatial_H,3));
 
figure
subplot(1,3,1)
imagesc(lat_sat(indexX(1):indexX(2)),lon_sat(indexY(1):indexY(2)),E_spatial_model_avg)
title('Model')
xlim(lat_sat([indexX(1) indexX(2)-1]))
ylim(lon_sat([indexY(1) indexY(2)-1]))
clim([min([E_spatial_model_avg E_spatial_H_avg],[],'all'),max([E_spatial_model_avg E_spatial_H_avg],[],'all')])
colormap turbo

set(gca,'Xdir','reverse','Ydir','reverse')
view(270,270)

subplot(1,3,2)
imagesc(lat_sat(indexX(1):indexX(2)),lon_sat(indexY(1):indexY(2)),E_spatial_H_avg)
title('RC compensated model')
xlim(lat_sat([indexX(1) indexX(2)-1]))
ylim(lon_sat([indexY(1) indexY(2)-1]))
clim([min([E_spatial_model_avg E_spatial_H_avg],[],'all'),max([E_spatial_model_avg E_spatial_H_avg],[],'all')])
colormap turbo

set(gca,'Xdir','reverse','Ydir','reverse')
view(270,270)

h = colorbar;
set(h,'position',[0.75,0.1,0.02,0.75])
h.Label.String = 'E';
h.Label.FontSize = 12;
