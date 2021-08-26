close all;
clc;

r=normrnd(0,2,[2000,1]);
theta=2*pi*rand([2000,1]);
sampleSize=100;
delay=0.05;

p=zeros([2000,3]);
p(1001:2000,3)=1;

%First cluster centred on point (5,10)
p(1:1000)=5+r(1:1000).*cos(theta(1:1000));
p(1:1000,2)=10+r(1:1000).*sin(theta(1:1000));

%Second cluster centred on point (10,5)
p(1001:2000)=10+r(1001:2000).*cos(theta(1001:2000));
p(1001:2000,2)=5+r(1001:2000).*sin(theta(1001:2000));

% Select random points for training
sp=randi([1,2000],sampleSize,1);

% Set initial weights and bias
w1=0.5;
w2=0.5;
b=0;

%Training%

for i=1:25

    % Calculate the output based on the set weights and bias
    y=w1*p(sp(i),1)+w2*p(sp(i),2)+b;
    
    % Limit the output to 1 or 0 since this is a classification problem
    if y<0
        y=0;
    else
        y=1;
    end
    
    % Calcuate the error and update weights and bias based on the error
    e=p(sp(i),3)-y;
    
    % Perceptron update rule
    w1=w1+e*p(sp(i),1);
    w2=w2+e*p(sp(i),2);
    b=b+e;
    
    pause(delay);
    clf;
    
    % Draw a line seperating the clusters based on the weights and bias
    input=linspace(0,20);
    output=(-b-(w1*input))/w2;
    
    hold on;
    xlim([0 20])
    ylim([0 20])
    text(10,10,int2str(i),'FontSize',20)
    scatter(p(1:1000),p(1:1000,2),5,'filled','blue')
    scatter(p(1001:2000),p(1001:2000,2),5,'filled','black')
    plot(input,output,'red','LineWidth',2)

    % Save plots in a Gif
    frame = getframe(1);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    if i == 1
        imwrite(imind,cm,"perceptron.gif",'gif', 'Loopcount',inf);
    else
        imwrite(imind,cm,"perceptron.gif",'gif','WriteMode','append');
    end
    
end
