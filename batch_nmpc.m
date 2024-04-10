%
% NMPC Batch reactor
clc; clear all; close all;
global Tref u_mpc Np k CA_mpc1 T_mpc1
ult_data = [];
for i=1:5
N=40;Tref=ones(N,1);
% for i=1:size(Tref,1)
% if (i-1)<=10/(80/N)
% Tref(i) = 298+0.5*(i-1)*(80/N);
% elseif (i-1)<=40/(80/N)
% Tref(i) = 303;
% elseif (i-1)<=50/(80/N)
% Tref(i) = 303+0.3*((i-1)*(80/N) - 40);
% elseif (i-1)<=60/(80/N)
% Tref(i) = 306;
% elseif (i-1)<=75/(80/N)
% Tref(i) = 306-(8/15)*((i-1)*(80/N)-60);
% else
% Tref(i)=298;
% end
% end

% Update values based on conditions
% Update values based on conditions
Tref(1:N/5) = randi([294, 307]);
Tref(N/5+1:3*N/5) = randi([294, 307]);
Tref(3*N/5+1:end) = randi([294, 307]);
plot(1:N,Tref,'r-');

T_mpc1 = 298; % T0
CA_mpc1 = 0.6 ; % CA0
T_mpc(1) = 298; % T0
CA_mpc(1) = 0.6 ; % CA0
u_mpc=295;
Np=10;

for k=1:N-1
    k;
if k+Np >= N
    Np = N-k;
else
    Np=10;
end
z0 = [ones(1,Np)*298, ones(1,Np)*0.5, ones(1,Np)*300];
options = optimset('Algorithm','interior-point','TolX',1e-15);
options.MaxFunEvals = 1e+07;
options.MaxIter = 1e+05;
LB(1:Np)=293;UB(1:Np)=308;
LB(Np+1:2*Np)=0;UB(Np+1:2*Np)=1;
LB((2*Np)+1:3*Np)=273;UB((2*Np)+1:3*Np)=318;

z = fmincon(@batch_obj,z0,[],[],[],[],LB,UB,@batch_cons,options)
u_mpc = z(2*Np+1);
Tj_mpc(k) = u_mpc;
T_mpc1 = z(1);
CA_mpc1 = z(Np+1);
T_mpc(k+1) = z(1);
CA_mpc(k+1) = z(Np+1);
end
hold on;
plot(1:N,T_mpc,'b-')
stairs(1:N-1,Tj_mpc,'g-')

figure(2)
plot(1:N,CA_mpc,'y-')
%% For DQN implementaion
error_next = [];
error = [];
error(1) = T_mpc(1) - Tref(1);
time_step = 0:N-1;
time_step_next = time_step(2:end);
time_step_next(N) = N-1;
T_mpc_next = T_mpc(2:end);
T_mpc_next(N) = 0;
CA_mpc_next = CA_mpc(2:end);
CA_mpc_next(N) = 0;
action = [];
for j = 1:N-1
action(j) = action_index(Tj_mpc(j));
end
action(N) = 0;
prev_action = 0;
d_action= [];
reward = [];
for j= 1:N
error_next(j) = T_mpc_next(j) - Tref(j);
end
for j = 2:N
error(j) = error_next(j-1);
end
for j = 1:N
    d_action(j) = action(j)-prev_action;
    reward(j) = -abs(error_next(j))-abs(d_action(j));
    prev_action = action(j);
end

data = [time_step', T_mpc', CA_mpc',error',action' ,time_step_next', T_mpc_next', CA_mpc_next',error_next',reward'];
ult_data = [ult_data;data];
end

T = array2table(ult_data, 'VariableNames', {'curr_ts', 'curr_T', 'curr_C','error','action', 'next_ts', 'next_T', 'next_C','error_next','reward'});

% Write table to Excel file
filename = 'Sample_data.xlsx';
writetable(T, filename);



function temp_index = action_index(action)
    % Calculates the index of action in action array 
    % Parameters:
    %     action: a scalar value

    num_temp = 40;
    tj_list = linspace(273, 318, num_temp);

    temp_delta = 0.5*(318 - 273) / (num_temp - 1);
    % finding action index with built in MATLAB function
    if 273 > action
        temp_index = 1;
    elseif 318 < action
        temp_index = num_temp;
    else
        indices = find(abs(tj_list - action) <= temp_delta);
        temp_index = indices(randi(length(indices)));
    end
end

function cost = batch_obj(z)
global Tref u_mpc Np k
T=z(1:Np);
CA=z(Np+1:2*Np);
Tj=z(2*Np+1:3*Np);
%Tj(1) = u_mpc;
eT=T-Tref(k:Np+k-1)';
eC = CA;
Q=[1 0; 0 1];R=0.1;
%uref=[299.827918341673	303.388595326598	305.276794262443	305.909183863504	305.442112236317	304.293211608289	303.466507809745	303.074595434477	302.955382130417	302.944454861204	302.944819487014	302.922364749299	302.899208510775	302.969663573633	303.322898143443	304.202283872710	305.654780257143	306.866690172651	307.527013379838	307.491452757883	306.877295696837	306.073935737908	304.797274109167	302.676451922803	300.192892749396	298.077877412719	296.464794815912	295.423571457608	295.110036241334	295.488520042256];
cost=0;
for i=1:Np
cost = cost+ [eT(i) eC(i)]*Q*[eT(i) eC(i)]' ;
end
for i = 1:Np
    if k==1
       cost = cost  ;%+ (0)*(Tj(i) - uref(i+k-1))^2+ (R)*(Tj(i) - 295)^2
    elseif i==1
    cost = cost  + (R)*(Tj(i) - u_mpc)^2; 
    else
    cost = cost + (R)*(Tj(i) - Tj(i-1))^2;
    end
end
cost;
end

function [Cineq,Ceq] = batch_cons(z)
global u_mpc Np CA_mpc1 T_mpc1
phi1= -0.09;
phi2= -1.64;
Ea_R= 13550;
k0=2.53*10^19;
% T0 = 300; % T0
% CA0 = 0.6 ; % CA0
Ts=80/39;
Ceq=[];
Cineq=[];
T=z(1:Np);
CA=z(Np+1:2*Np);
Tj=z(2*Np+1:3*Np);
T0 = T_mpc1; % T0
CA0 = CA_mpc1; % CA0
Tj(1) = u_mpc;
for i=1:Np
ineq_cons=[];
Cineq=[Cineq;ineq_cons];
end
Ceq=[T(1) - T0 - Ts*(phi1*(T0-Tj(1))+phi2*k0*exp(-Ea_R/T0)*CA0^2);
CA(1) - CA0 - Ts*(-k0*exp(-Ea_R/T0)*CA0^2)];
for i=2:Np
cons=[CA(i) - CA(i-1) - Ts*(-k0*exp(-Ea_R/T(i-1))*(CA(i-1))^2);
T(i) - T(i-1) - Ts*(phi1*(T(i-1)-Tj(i-1))+phi2*k0*exp(-Ea_R/T(i-1))*CA(i-1)^2);];
Ceq=[Ceq;cons];
end
end
