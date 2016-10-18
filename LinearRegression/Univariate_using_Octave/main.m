clear ; close all; clc;

%Load Car Data
%weight,mpg - data.txt ( 100 rows )
car_data=load('data.txt');

x=car_data(:,1); % X would have 100 cols & 1 row
y=car_data(:,2); % y would have 100 cols & 1 row

m=length(y);

%Now lets plot the data in scatter plot
figure;
plot(x,y,'bx','MarkerSize',10);
xlabel('(y) Weight -->','fontsize',14);
ylabel('(x) mpg -->','fontsize',14);

fprintf('Press any key to continue ...');
pause;

%now we will calculate Gradient Descent
%add x0 to the feature matrix
x=[ones(m,1),x];

%initialize theta θ=[θ0,θ1]
theta=zeros(2,1);

num_of_iterations = 1000;
alpha = 0.1;

J=zeros(num_of_iterations,1);

for i=1:num_of_iterations

  h_of_x=(x*theta).-y;
  
  %h_of_x so that we would get a real number ( element wise multiplication + sum )
 
  theta(1)=theta(1)-(alpha/m)*h_of_x'*x(:,1);
  theta(2)=theta(2)-(alpha/m)*h_of_x'*x(:,2);
 
  %compute J(θ) - Cost
  J(i)=1/(2*m)*sum(h_of_x.^2);
  
end

fprintf('θ0 = %f θ1 = %f \n', theta(1), theta(2));

hold on;

%First plot x, the for y calculate y. 
%(100X2) * ( 2 X 1) =( 100 X 1 )
plot(x(:,2), x*theta, 'r-','linewidth',2);

hold off;

fprintf('Press any key to continue ...');
pause;

predict=[1.3,3.6,5.5];

hold on;

for i=predict

plot(i, [1, i]*theta, 'g*','MarkerSize',14);
legend('Training data', 'Linear regression','Prediction')

end

hold off;

fprintf('Press any key to continue ...');
pause;

thetaNormal = (pinv(x'*x))*x'*y;

hold on;

plot(x(:,2), x*thetaNormal, 'k-','linewidth',1);
hold off;

fprintf('Press any key to continue ...');
pause;

figure;

plot([1:num_of_iterations],J,'linewidth',2);
xlabel('Num of iterations -->','fontsize',14);
ylabel('Cost Function J(theta) -->','fontsize',14);

