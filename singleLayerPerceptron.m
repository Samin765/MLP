
clf;

%parameters
runs = 25;
epochs = 100;
n = 100;
anim = false;
seperable = false;
divided = true;
remove_from_A = 0;
remove_from_B = 0;
skewed_sampling = true; % 0% from a subset of classA for which classA(1,:)<0 and 80% from a subset of classA for which classA(1,:)>0

%Generate datapoints
if divided
    % In this case, classA (Blue) is divided into two regions
    mA = [ 1.0, 0.3]; sigmaA = 0.2;
    mB = [ 0.0, -0.1]; sigmaB = 0.3;
    classA(1,:) = [ randn(1,round(0.5*n)) .* sigmaA - mA(1), ...
    randn(1,round(0.5*n)) .* sigmaA + mA(1)];
    classA(2,:) = randn(1,n) .* sigmaA + mA(2);
    classB(1,:) = randn(1,n) .* sigmaB + mB(1);
    classB(2,:) = randn(1,n) .* sigmaB + mB(2);
else
    if seperable
        % Make mA is far from mB
        mA = [ 6.0, 6.0]; sigmaA = 1.5;
        mB = [0, 0]; sigmaB = 1;
    else
        % Make mA close to mB
        mA = [ 2.0, 2.0]; sigmaA = 1.5;
        mB = [0, 0]; sigmaB = 1;
    end
    classA(1,:) = randn(1,n) .* sigmaA + mA(1);
    classA(2,:) = randn(1,n) .* sigmaA + mA(2);
    classB(1,:) = randn(1,n) .* sigmaB + mB(1);
    classB(2,:) = randn(1,n) .* sigmaB + mB(2);
end

classA(3,:) = -1;
classB(3,:) = 1;
data = cat(2, classA, classB);
X_all = [data(1,:);data(2, :); ones(1, 2*n)];
T_all = data(3, :);

if skewed_sampling
    R = rand(1,n);
    sampleA = classA(:, R < 0.2 | (classA(1, :) < 0 & R < 0.8));
    size(sampleA)
    sampleB = classB;
else
    sampleA = classA(:, 1:floor((1-remove_from_A)*n));
    sampleB = classB(:, 1:floor((1-remove_from_B)*n));
end
sample = cat(2, sampleA, sampleB);
s = size(sample, 2); % Number of samples (approximately 75% or 100% of 2*n)
sample = sample(:, randperm(s, s));
X_sample = [sample(1,:); sample(2, :); ones(1, s)];
T_sample = sample(3, :);

% Variables that contain statistical information
final_score = zeros(3, runs);
score_history = zeros(3, epochs);
W_result = zeros(3, 3);

eta = 0.0005;
% Sequential perceptron rule
for run = 1:runs
    W = randn(1,3);
    for i = 1: epochs
        for j = 1:s
            dataPoint = sample(:, j);
            X = [dataPoint(1);dataPoint(2); 1];
            T = dataPoint(3);
            O = sign(W*X);
            W = W + eta * (T - O) * X';
        end
        if anim & O ~= T
            plotnow(data, colors, W);
        end
        score_history(1, i) = score_history(1, i) + getScore(n, W, X_all, T_all)/runs;
    end
    finalScores(1, run) = getScore(n, W, X_all, T_all);
end

W_result(1, :) = W;
disp("Perceptron rule (" + string(epochs) + " epochs)");
disp("Average score: " + string(mean(finalScores(1, :))));
disp("Standard deviation: " + string(std(finalScores(1, :))));

% Sequential delta rule

eta = 0.005;
for run = 1:runs
    W = randn(1,3);
    for i = 1:epochs
        for j = 1:s
            dataPoint = sample(:, j);
            X = [dataPoint(1);dataPoint(2); 1];
            T = dataPoint(3);
            W = W + eta * (T - W * X) * X'; 
        end
        score_history(2, i) = score_history(2, i) + getScore(n, W, X_all, T_all)/runs;
        if anim
            plotnow(data, n, W);
        end 
    end
    finalScores(2, run) = getScore(n, W, X_all, T_all);
end

W_result(2, :) = W;
disp("Sequential delta rule (" + string(epochs) + " epochs)");
disp("Average score: " + string(mean(finalScores(2, :))));
disp("Standard deviation: " + string(std(finalScores(2, :))));

% Batch delta rule

eta = 0.1/n;
for run = 1:runs
    W = randn(1,3);
    for i = 1:epochs
        W = W + eta * (T_sample - W * X_sample) * X_sample';
        O = sign(W*X_sample);
        score_history(3, i) = score_history(3, i) + getScore(n, W, X_all, T_all)/runs;
        if anim & O ~= T
            plotnow(data, colors, W);
        end
    end
    finalScores(3, run) = getScore(n, W, X_all, T_all);
end

W_result(3, :) = W;
disp("Batch delta rule (" + string(epochs) + " epochs)");
disp("Average score: " + string(mean(finalScores(3,:))));
disp("Standard deviation: " + string(std(finalScores(3, :))));

% Display the data and the three lines
figure(1);
clf;
x_range = -4 : 13;
y_range = -(W_result(:,3) + W_result(:,1) .* x_range) ./ W_result(:,2);

for i = 1 : 3
    plot(x_range, y_range(i, :));
    hold on;
end
hold on;
scatter(classA(1,:), classA(2,:), "blue");
hold on;
scatter(classB(1, :), classB(2, :), "red");
axis([-4, -4; 13, 13]);
method_names = {'perceptron', 'delta rule', 'batch delta rule'};
title("Noneperable dataset - Decision boundaries");
legend(method_names, 'Location','southeast');

% Plot learning curves
figure(2);
clf;
for i = 1:3
    plot(1:epochs, score_history(i,:));
    hold on;
end
title("Noneperable dataset - Learning curves")
legend(method_names, 'Location','southeast');
xlabel("batch");
ylabel("accuracy");

% Used for animation:
function plotnow(data, colors, W)
    x_range = -4:10;
    y_range = -(W(3)+W(1)*x_range)/W(2);
    clf;
    plot(x_range, y_range);
    hold on;
    scatter(data(1,:), data(2,:),[], colors);
    drawnow
end

function score = getScore(n, W, X, T)
    O = sign(W*X);
    score = 0;
    for j = 1:2*n
        if O(j) == T(j)
            score = score + 1;
        end
    end
    score = score / (2*n);
end
