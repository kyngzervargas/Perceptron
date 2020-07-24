clear all;
close all;

load apple(hue,ecc).mat
load orange(hue,ecc).mat
load banana(hue,ecc).mat

%% 
no_ = ones(1,size(apple,1))*-1;
% no_ = zeros(1,size(apple,1)); %FOR ACT 14
% no_ = ones(1,size(banana,1))*-1;
o_ = ones(1,size(orange,1));
no_ = no_';
o_ = o_';

%LABEL matrix
d = [o_;no_];
% d = flipud(d);

%Data matrix
data = [apple;orange];
% data = [banana;orange];

%Variables
bias = ones(1,size(data,1));
bias = bias';
n = 0.5; %learning rate

%WEIGHTS
a_ = 0; b_ = 1; %range
w = (b_-a_).*rand(3,1)/100 + a_;
w = w';
% weights = -1*2.*rand(3,1);

X = [bias,data]; %Data

%% FOR ACT 13
maxIters = 100;
eps_ = [];
z_ = [];
aa = [];
res = [];
SSE = 1;
error = 0.01;
count = 0;

for j = 1:maxIters
% while abs(SSE) > error
for i = 1:length(X)
% for i = 1:22
    a = (X(i,1).*w(1)) + (X(i,2).*w(2)) + (X(i,3).*w(3)); 
    if a >= 0;
        z = 1;
    else 
        z = -1;
    end
    d_ = d(i); 
    eps = (d_-z);
    delta = n* eps.*X(i,:);
%     predictions  = [predictions; delta];
    w = w + delta;
    
    %stores the values
    aa = [aa;a];
    z_ = [z_;z];
    eps_ = [eps_;eps];   
    res = [res;(eps^2)];
    
end
    SSE = sum(res);
    eps_ = [];
    z_ = [];
    aa = [];
    res = [];
    if SSE <= error
        disp('done');
        break
% %     else
% %         maxIters = maxIters + 1;
    end
%     count = count + 1;
%     disp(count);
%     if count == 50
%         break 
%     end
% end
end

%% FOR ACT 14
% maxIters = 100;
% eps_ = [];
% z_ = [];
% aa = [];
% res = [];
% SSE = 1;
% error = 0.01;
% count = 0;
% 
% for j = 1:maxIters
% % while abs(SSE) > error
% for i = 1:length(X)
% % for i = 1:22
%     a = (X(i,1).*w(1)) + (X(i,2).*w(2)) + (X(i,3).*w(3)); 
% %     if a >= 0;
% %         z = 1;
% %     else 
% %         z = -1;
% %     end
% 
% %Logistic Function
%     z = 1/(1+exp(-a));
% 
%     d_ = d(i); 
%     eps = (d_-z);
%     delta = n* eps.*X(i,:);
% %     predictions  = [predictions; delta];
%     w = w + delta;
%     
%     %stores the values
%     aa = [aa;a];
%     z_ = [z_;z];
%     eps_ = [eps_;eps];   
%     res = [res;(eps^2)];
%     
% end
%     SSE = sum(res);
%     eps_ = [];
%     z_ = [];
%     aa = [];
%     res = [];
%     if SSE <= error
%         disp('done');
%         break
% % %     else
% % %         maxIters = maxIters + 1;
%     end
% %     count = count + 1;
% %     disp(count);
% %     if count == 50
% %         break 
% %     end
% % end
% end

%% GRAPH
figure(1);
scatter(apple(:,1),apple(:,2),'x');
hold on;
scatter(orange(:,1),orange(:,2),'*');
scatter(banana(:,1),banana(:,2),'d');


xx = linspace(min(X(:,2)),max(X(:,2)));
A = w(2);
B = w(3);
C = -w(1);

m = -A/B;
bb = C/B;
yy = m*xx + bb;

plot(xx,yy);
% legend('apple','orange', 'banana','decision line', 'Location', 'best');
legend('apple','orange', 'banana', 'Location', 'best');

y_min = min(X(:,3))-(max(X(:,3))/5);
y_max = max(X(:,3))+(max(X(:,3))/5);
ylim([y_min y_max]);
% ylim([-2 2]);

title('Feature Extraction');
% title('Perceptron');
xlabel('Hue');
ylabel('Eccentricity');
%%
% function z = g(a)
% if  a >= 0
%     z = 1;
% else 
%     z = -1;
% end
% end

%%

% input = [0 0; 0 1; 1 0; 1 1];
% numIn = 4;
% desired_out = [0;1;1;1];
% bias = 1;
% coeff = 0.7;
% rand('state',sum(100*clock));
% weights = -1*2.*rand(3,1);
% 
% iterations = 10;
% 
% for i = 1:iterations
%      out = zeros(4,1);
%      for j = 1:numIn
%           y = bias*weights(1,1)+...
%                input(j,1)*weights(2,1)+input(j,2)*weights(3,1);
%           out(j) = 1/(1+exp(-y));
%           delta = desired_out(j)-out(j);
%           weights(1,1) = weights(1,1)+coeff*bias*delta;
%           weights(2,1) = weights(2,1)+coeff*input(j,1)*delta;
%           weights(3,1) = weights(3,1)+coeff*input(j,2)*delta;
%      end
% end